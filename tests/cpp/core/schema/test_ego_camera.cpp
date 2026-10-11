// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/idl.h>
#include <schema/ego_audio_generated.h>
#include <schema/ego_calibration_generated.h>
#include <schema/ego_camera_generated.h>
#include <schema/ego_device_state_bfbs_generated.h>
#include <schema/ego_device_state_utils.hpp>
#include <schema/ego_imu_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>
#include <schema_compat/schema_compat.hpp>

#include <cstdint>
#include <memory>

#define VT(field) (field + 2) * 2

static_assert(core::EgoFrameMetadata::VT_STREAM == VT(0));
static_assert(core::EgoFrameMetadata::VT_SEQUENCE_NUMBER == VT(1));
static_assert(core::EgoFrameMetadata::VT_WIDTH == VT(2));
static_assert(core::EgoFrameMetadata::VT_HEIGHT == VT(3));
static_assert(core::EgoFrameMetadata::VT_FPS == VT(4));
static_assert(core::EgoFrameMetadata::VT_PIXEL_FORMAT == VT(5));
static_assert(core::EgoFrameMetadata::VT_ENCODED_BYTES == VT(6));
static_assert(core::EgoFrameMetadata::VT_SDK_METADATA == VT(7));
static_assert(core::EgoFrameMetadata::VT_CAPTURE_EPOCH == VT(8));
static_assert(core::EgoPixelFormat_Mjpg == 0);
static_assert(core::EgoAudioChunk::VT_TIMING == VT(9));
static_assert(core::EgoPcmAudioChunk::VT_TIMING == VT(8));
static_assert(core::EgoCalibration::VT_TIME_ALIGNMENT == VT(11));

TEST_CASE("EGO timing tables preserve optional provenance through record serialization", "[unit][ego][schema]")
{
    core::EgoCalibrationT calibration;
    CHECK(core::pack<core::EgoCalibration>(calibration)->time_alignment() == nullptr);
    calibration.time_alignment = std::make_shared<core::EgoTimeAlignmentT>();
    calibration.time_alignment->json = "{\"version\":1,\"status\":\"unverified\"}";
    const auto encoded = core::pack<core::EgoCalibration>(calibration);
    REQUIRE(encoded->time_alignment());
    CHECK(encoded->time_alignment()->json()->str() == calibration.time_alignment->json);

    core::EgoAudioChunkT audio;
    CHECK(core::pack<core::EgoAudioChunk>(audio)->timing() == nullptr);
    audio.timing = std::make_shared<core::EgoAudioTimingT>();
    audio.timing->segment_id = 42;
    audio.timing->first_sample_index = 1ULL << 40;
    audio.timing->discontinuity = true;
    // Schema tests use native records; pack_record belongs to the separate MCAP module.
    core::EgoAudioChunkRecordT native_record;
    native_record.data = std::make_shared<core::EgoAudioChunkT>(audio);
    native_record.timestamp = std::make_shared<core::DeviceDataTimestamp>(30, 20, 10);
    const auto record = core::pack<core::EgoAudioChunkRecord>(native_record);
    REQUIRE(record->data()->timing());
    CHECK(record->data()->timing()->segment_id() == 42);
    CHECK(record->data()->timing()->first_sample_index() == (1ULL << 40));
    CHECK_FALSE(record->data()->timing()->continuity_known());
    CHECK(record->data()->timing()->discontinuity());
    CHECK(record->timestamp()->sample_time_raw_device_clock() == 10);

    core::EgoPcmAudioChunkT pcm;
    pcm.timing = audio.timing;
    const auto pcm_encoded = core::pack<core::EgoPcmAudioChunk>(pcm);
    REQUIRE(pcm_encoded->timing());
    CHECK(pcm_encoded->timing()->first_sample_index() == audio.timing->first_sample_index);
}

TEST_CASE("EGO camera metadata round trips", "[unit][ego][schema]")
{
    core::EgoFrameMetadataT original;
    original.stream = core::EgoCameraStream_ColorRight;
    original.sequence_number = UINT64_MAX;
    original.width = 1280;
    original.height = 720;
    original.fps = 30;
    original.pixel_format = core::EgoPixelFormat_Mjpg;
    original.encoded_bytes = 1234;
    original.sdk_metadata.emplace_back(1, 99);
    original.capture_epoch = 3;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::EgoFrameMetadata::Pack(builder, &original));

    core::EgoFrameMetadataT restored;
    flatbuffers::GetRoot<core::EgoFrameMetadata>(builder.GetBufferPointer())->UnPackTo(&restored);
    CHECK(restored.stream == core::EgoCameraStream_ColorRight);
    CHECK(restored.sequence_number == UINT64_MAX);
    CHECK(restored.width == 1280);
    CHECK(restored.height == 720);
    CHECK(restored.fps == 30);
    CHECK(restored.pixel_format == core::EgoPixelFormat_Mjpg);
    CHECK(restored.encoded_bytes == 1234);
    REQUIRE(restored.sdk_metadata.size() == 1);
    CHECK(restored.sdk_metadata[0].value() == 99);
    CHECK(restored.capture_epoch == 3);
}

TEST_CASE("EGO auxiliary schemas round trip", "[unit][ego][schema]")
{
    core::EgoImuBatchT imu;
    imu.sensor = core::EgoImuSensor_Gyro;
    imu.sequence_number = 5;
    imu.sample_rate_hz = 1000;
    imu.full_scale = 500;
    imu.samples.emplace_back(1, 2, 3, 24, 100, 200);
    imu.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder imu_builder;
    imu_builder.Finish(core::EgoImuBatch::Pack(imu_builder, &imu));
    core::EgoImuBatchT restored_imu;
    flatbuffers::GetRoot<core::EgoImuBatch>(imu_builder.GetBufferPointer())->UnPackTo(&restored_imu);
    REQUIRE(restored_imu.samples.size() == 1);
    CHECK(restored_imu.samples[0].z_si() == 3);
    CHECK(restored_imu.capture_epoch == 2);

    core::EgoAudioChunkT audio;
    audio.sample_rate_hz = 48000;
    audio.channel_count = 1;
    audio.bits_per_sample = 16;
    audio.sample_count = 480;
    audio.wav_data_offset = 44;
    audio.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder audio_builder;
    audio_builder.Finish(core::EgoAudioChunk::Pack(audio_builder, &audio));
    core::EgoAudioChunkT restored_audio;
    flatbuffers::GetRoot<core::EgoAudioChunk>(audio_builder.GetBufferPointer())->UnPackTo(&restored_audio);
    CHECK(restored_audio.wav_data_offset == 44);
    CHECK(restored_audio.sample_count == 480);
    CHECK(restored_audio.capture_epoch == 2);

    core::EgoCalibrationT calibration;
    calibration.device_uid = "ego";
    calibration.raw_alignment_yaml = "camera: left";
    calibration.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder calibration_builder;
    calibration_builder.Finish(core::EgoCalibration::Pack(calibration_builder, &calibration));
    core::EgoCalibrationT restored_calibration;
    flatbuffers::GetRoot<core::EgoCalibration>(calibration_builder.GetBufferPointer())->UnPackTo(&restored_calibration);
    CHECK(restored_calibration.device_uid == "ego");
    CHECK(restored_calibration.capture_epoch == 2);
}

TEST_CASE("EGO camera record retains timestamp", "[unit][ego][schema]")
{
    auto record = std::make_shared<core::EgoFrameMetadataRecordT>();
    record->data = std::make_shared<core::EgoFrameMetadataT>();
    record->timestamp = std::make_shared<core::DeviceDataTimestamp>(1, 2, 3);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::EgoFrameMetadataRecord::Pack(builder, record.get()));
    const auto* restored = flatbuffers::GetRoot<core::EgoFrameMetadataRecord>(builder.GetBufferPointer());
    REQUIRE(restored->timestamp() != nullptr);
    CHECK(restored->timestamp()->available_time_local_common_clock() == 1);
    CHECK(restored->timestamp()->sample_time_local_common_clock() == 2);
    CHECK(restored->timestamp()->sample_time_raw_device_clock() == 3);
}

TEST_CASE("EGO embedded media schemas round trip", "[unit][ego][schema]")
{
    core::EgoEncodedVideoFrameT video;
    video.stream = core::EgoCameraStream_ColorRight;
    video.sequence_number = 42;
    video.width = 1600;
    video.height = 1300;
    video.fps = 30;
    video.pixel_format = core::EgoPixelFormat_H264;
    video.encoded_data = { 0, 0, 0, 1, 0x65, 1 };
    video.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder video_builder;
    video_builder.Finish(core::EgoEncodedVideoFrame::Pack(video_builder, &video));
    core::EgoEncodedVideoFrameT restored_video;
    flatbuffers::GetRoot<core::EgoEncodedVideoFrame>(video_builder.GetBufferPointer())->UnPackTo(&restored_video);
    CHECK(restored_video.stream == core::EgoCameraStream_ColorRight);
    CHECK(restored_video.encoded_data == video.encoded_data);
    CHECK(restored_video.capture_epoch == 2);

    core::EgoPcmAudioChunkT audio;
    audio.sequence_number = 7;
    audio.sample_rate_hz = 48000;
    audio.channel_count = 1;
    audio.bits_per_sample = 16;
    audio.sample_count = 2;
    audio.pcm_data = { 1, 0, 2, 0 };
    audio.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder audio_builder;
    audio_builder.Finish(core::EgoPcmAudioChunk::Pack(audio_builder, &audio));
    core::EgoPcmAudioChunkT restored_audio;
    flatbuffers::GetRoot<core::EgoPcmAudioChunk>(audio_builder.GetBufferPointer())->UnPackTo(&restored_audio);
    CHECK(restored_audio.pcm_data == audio.pcm_data);
    CHECK(restored_audio.capture_epoch == 2);
}

TEST_CASE("EGO diagnostic events preserve typed payloads and optional zero", "[unit][ego][schema]")
{
    core::EgoDeviceStateT native;
    native.diagnostic_version = core::EGO_DIAGNOSTIC_VERSION;
    native.device_uid = "test-device";
    native.capture_epoch = 2;
    native.observed_at_ns = 100;
    SECTION("SDK report keeps a real zero error clear")
    {
        core::EgoSdkReportT report;
        report.sequence_number = UINT64_MAX;
        report.error_flags = 0;
        native.event.Set(report);
        const auto state = core::pack<core::EgoDeviceState>(native);
        CHECK_NOTHROW(core::validate_ego_diagnostic(*state));
        CHECK(state->event_as_EgoSdkReport()->sequence_number() == UINT64_MAX);
        CHECK(state->event_as_EgoSdkReport()->error_flags() == 0);
        CHECK(state->event_as_EgoRecoveryEvent() == nullptr);
    }
    SECTION("final summary keeps cumulative losses")
    {
        core::EgoCaptureSummaryT summary;
        summary.capture_health = core::EgoCaptureHealth_Incomplete;
        summary.reason = "dropped video";
        summary.queue_capacity = 4096;
        summary.queue_peak = 99;
        summary.dropped_video_frame_sets = 3;
        summary.is_final = true;
        native.event.Set(summary);
        auto state = core::pack<core::EgoDeviceState>(native);
        CHECK_NOTHROW(core::validate_ego_diagnostic(*state));
        auto retained = state.narrow(state->event_as_EgoCaptureSummary());
        state.reset();
        CHECK(retained->dropped_video_frame_sets() == 3);
        CHECK(retained->is_final());
        CHECK(retained->reason()->str() == "dropped video");
    }
    SECTION("recovery carries only its transition")
    {
        core::EgoRecoveryEventT recovery;
        recovery.stage = core::EgoRecoveryStage_Recovered;
        recovery.reconnect_attempt = 4;
        native.event.Set(recovery);
        const auto state = core::pack<core::EgoDeviceState>(native);
        CHECK_NOTHROW(core::validate_ego_diagnostic(*state));
        CHECK(state->capture_epoch() == 2);
        CHECK(state->event_as_EgoRecoveryEvent()->reconnect_attempt() == 4);
    }
    SECTION("unavailable and zero temperature differ on the wire")
    {
        core::EgoDeviceSnapshotT snapshot;
        snapshot.properties.emplace_back(279, 0);
        native.event.Set(snapshot);
        const auto missing = core::pack<core::EgoDeviceState>(native);
        CHECK_NOTHROW(core::validate_ego_diagnostic(*missing));
        CHECK_FALSE(missing->event_as_EgoDeviceSnapshot()->imu_temperature_c().has_value());
        snapshot.imu_temperature_c = 0.0f;
        native.event.Set(snapshot);
        const auto measured = core::pack<core::EgoDeviceState>(native);
        const auto temperature = measured->event_as_EgoDeviceSnapshot()->imu_temperature_c();
        REQUIRE(temperature.has_value());
        CHECK(*temperature == 0.0f);
        CHECK(measured->event_as_EgoDeviceSnapshot()->properties()->Get(0)->value() == 0);
    }
}

TEST_CASE("EGO diagnostics reject missing events and unsupported versions", "[unit][ego][schema]")
{
    core::EgoDeviceStateT native;
    native.device_uid = "test-device";
    native.observed_at_ns = 100;
    native.diagnostic_version = core::EGO_DIAGNOSTIC_VERSION;
    CHECK_THROWS_AS(core::validate_ego_diagnostic(*core::pack<core::EgoDeviceState>(native)), std::invalid_argument);
    native.event.Set(core::EgoSdkReportT{});
    native.diagnostic_version = 1;
    CHECK_THROWS_AS(core::validate_ego_diagnostic(*core::pack<core::EgoDeviceState>(native)), std::invalid_argument);
    native.diagnostic_version = core::EGO_DIAGNOSTIC_VERSION;
    auto state = core::pack<core::EgoDeviceState>(native);
    auto bytes = std::vector<uint8_t>(state.buffer().begin(), state.buffer().end());
    auto* mutable_state = flatbuffers::GetMutableRoot<core::EgoDeviceState>(bytes.data());
    // The generated verifier accepts unknown union tags; the EGO consumer rejects them.
    auto* table = reinterpret_cast<flatbuffers::Table*>(mutable_state);
    REQUIRE(table->SetField<uint8_t>(core::EgoDeviceState::VT_EVENT_TYPE, 99));
    CHECK_THROWS_AS(core::validate_ego_diagnostic(*mutable_state), std::invalid_argument);
}

TEST_CASE("EGO event protocol intentionally rejects the former diagnostic layout", "[unit][ego][schema]")
{
    flatbuffers::Parser old;
    REQUIRE(old.Parse(R"(
        namespace core;
        table EgoDeviceState { sequence_number: uint64 (id: 0); device_uid: string (id: 1); }
        table EgoDeviceStateRecord { data: EgoDeviceState (id: 0); }
        root_type EgoDeviceStateRecord;
    )"));
    old.Serialize();
    const auto current = core::EgoDeviceStateRecordBinarySchema::data();
    const auto result = core::check_schema_compat({ old.builder_.GetBufferPointer(), old.builder_.GetSize() },
                                                  { current, core::EgoDeviceStateRecordBinarySchema::size() });
    CHECK(result.status == core::SchemaCompat::Incompatible);
}
