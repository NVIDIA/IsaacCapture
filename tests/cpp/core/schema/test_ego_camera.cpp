// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/idl.h>
#include <schema/ego_device_state_utils.hpp>
#include <schema/orbbec_ego_audio_generated.h>
#include <schema/orbbec_ego_calibration_generated.h>
#include <schema/orbbec_ego_camera_generated.h>
#include <schema/orbbec_ego_device_state_bfbs_generated.h>
#include <schema/orbbec_ego_imu_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>
#include <schema_compat/schema_compat.hpp>

#include <cstdint>
#include <memory>

#define VT(field) (field + 2) * 2

static_assert(core::OrbbecEgoFrameMetadata::VT_STREAM == VT(0));
static_assert(core::OrbbecEgoFrameMetadata::VT_SEQUENCE_NUMBER == VT(1));
static_assert(core::OrbbecEgoFrameMetadata::VT_WIDTH == VT(2));
static_assert(core::OrbbecEgoFrameMetadata::VT_HEIGHT == VT(3));
static_assert(core::OrbbecEgoFrameMetadata::VT_FPS == VT(4));
static_assert(core::OrbbecEgoFrameMetadata::VT_PIXEL_FORMAT == VT(5));
static_assert(core::OrbbecEgoFrameMetadata::VT_ENCODED_BYTES == VT(6));
static_assert(core::OrbbecEgoFrameMetadata::VT_SDK_METADATA == VT(7));
static_assert(core::OrbbecEgoFrameMetadata::VT_CAPTURE_EPOCH == VT(8));
static_assert(core::OrbbecEgoPixelFormat_Mjpg == 0);
static_assert(core::OrbbecEgoAudioChunk::VT_TIMING == VT(9));
static_assert(core::OrbbecEgoPcmAudioChunk::VT_TIMING == VT(8));
static_assert(core::OrbbecEgoCalibration::VT_TIME_ALIGNMENT == VT(11));

TEST_CASE("EGO timing tables preserve optional provenance through record serialization", "[unit][ego][schema]")
{
    core::OrbbecEgoCalibrationT calibration;
    CHECK(core::pack<core::OrbbecEgoCalibration>(calibration)->time_alignment() == nullptr);
    calibration.time_alignment = std::make_shared<core::OrbbecEgoTimeAlignmentT>();
    calibration.time_alignment->json = "{\"version\":1,\"status\":\"unverified\"}";
    const auto encoded = core::pack<core::OrbbecEgoCalibration>(calibration);
    REQUIRE(encoded->time_alignment());
    CHECK(encoded->time_alignment()->json()->str() == calibration.time_alignment->json);

    core::OrbbecEgoAudioChunkT audio;
    CHECK(core::pack<core::OrbbecEgoAudioChunk>(audio)->timing() == nullptr);
    audio.timing = std::make_shared<core::OrbbecEgoAudioTimingT>();
    audio.timing->segment_id = 42;
    audio.timing->first_sample_index = 1ULL << 40;
    audio.timing->discontinuity = true;
    // Schema tests use native records; pack_record belongs to the separate MCAP module.
    core::OrbbecEgoAudioChunkRecordT native_record;
    native_record.data = std::make_shared<core::OrbbecEgoAudioChunkT>(audio);
    native_record.timestamp = std::make_shared<core::DeviceDataTimestamp>(30, 20, 10);
    const auto record = core::pack<core::OrbbecEgoAudioChunkRecord>(native_record);
    REQUIRE(record->data()->timing());
    CHECK(record->data()->timing()->segment_id() == 42);
    CHECK(record->data()->timing()->first_sample_index() == (1ULL << 40));
    CHECK_FALSE(record->data()->timing()->continuity_known());
    CHECK(record->data()->timing()->discontinuity());
    CHECK(record->timestamp()->sample_time_raw_device_clock() == 10);

    core::OrbbecEgoPcmAudioChunkT pcm;
    pcm.timing = audio.timing;
    const auto pcm_encoded = core::pack<core::OrbbecEgoPcmAudioChunk>(pcm);
    REQUIRE(pcm_encoded->timing());
    CHECK(pcm_encoded->timing()->first_sample_index() == audio.timing->first_sample_index);
}

TEST_CASE("EGO camera metadata round trips", "[unit][ego][schema]")
{
    core::OrbbecEgoFrameMetadataT original;
    original.stream = core::OrbbecEgoCameraStream_ColorRight;
    original.sequence_number = UINT64_MAX;
    original.width = 1280;
    original.height = 720;
    original.fps = 30;
    original.pixel_format = core::OrbbecEgoPixelFormat_Mjpg;
    original.encoded_bytes = 1234;
    original.sdk_metadata.emplace_back(1, 99);
    original.capture_epoch = 3;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::OrbbecEgoFrameMetadata::Pack(builder, &original));

    core::OrbbecEgoFrameMetadataT restored;
    flatbuffers::GetRoot<core::OrbbecEgoFrameMetadata>(builder.GetBufferPointer())->UnPackTo(&restored);
    CHECK(restored.stream == core::OrbbecEgoCameraStream_ColorRight);
    CHECK(restored.sequence_number == UINT64_MAX);
    CHECK(restored.width == 1280);
    CHECK(restored.height == 720);
    CHECK(restored.fps == 30);
    CHECK(restored.pixel_format == core::OrbbecEgoPixelFormat_Mjpg);
    CHECK(restored.encoded_bytes == 1234);
    REQUIRE(restored.sdk_metadata.size() == 1);
    CHECK(restored.sdk_metadata[0].value() == 99);
    CHECK(restored.capture_epoch == 3);
}

TEST_CASE("EGO auxiliary schemas round trip", "[unit][ego][schema]")
{
    core::OrbbecEgoImuBatchT imu;
    imu.sensor = core::OrbbecEgoImuSensor_Gyro;
    imu.sequence_number = 5;
    imu.sample_rate_hz = 1000;
    imu.full_scale = 500;
    imu.samples.emplace_back(1, 2, 3, 24, 100, 200);
    imu.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder imu_builder;
    imu_builder.Finish(core::OrbbecEgoImuBatch::Pack(imu_builder, &imu));
    core::OrbbecEgoImuBatchT restored_imu;
    flatbuffers::GetRoot<core::OrbbecEgoImuBatch>(imu_builder.GetBufferPointer())->UnPackTo(&restored_imu);
    REQUIRE(restored_imu.samples.size() == 1);
    CHECK(restored_imu.samples[0].z_si() == 3);
    CHECK(restored_imu.capture_epoch == 2);

    core::OrbbecEgoAudioChunkT audio;
    audio.sample_rate_hz = 48000;
    audio.channel_count = 1;
    audio.bits_per_sample = 16;
    audio.sample_count = 480;
    audio.wav_data_offset = 44;
    audio.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder audio_builder;
    audio_builder.Finish(core::OrbbecEgoAudioChunk::Pack(audio_builder, &audio));
    core::OrbbecEgoAudioChunkT restored_audio;
    flatbuffers::GetRoot<core::OrbbecEgoAudioChunk>(audio_builder.GetBufferPointer())->UnPackTo(&restored_audio);
    CHECK(restored_audio.wav_data_offset == 44);
    CHECK(restored_audio.sample_count == 480);
    CHECK(restored_audio.capture_epoch == 2);

    core::OrbbecEgoCalibrationT calibration;
    calibration.device_uid = "ego";
    calibration.raw_alignment_yaml = "camera: left";
    calibration.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder calibration_builder;
    calibration_builder.Finish(core::OrbbecEgoCalibration::Pack(calibration_builder, &calibration));
    core::OrbbecEgoCalibrationT restored_calibration;
    flatbuffers::GetRoot<core::OrbbecEgoCalibration>(calibration_builder.GetBufferPointer())->UnPackTo(&restored_calibration);
    CHECK(restored_calibration.device_uid == "ego");
    CHECK(restored_calibration.capture_epoch == 2);
}

TEST_CASE("EGO camera record retains timestamp", "[unit][ego][schema]")
{
    auto record = std::make_shared<core::OrbbecEgoFrameMetadataRecordT>();
    record->data = std::make_shared<core::OrbbecEgoFrameMetadataT>();
    record->timestamp = std::make_shared<core::DeviceDataTimestamp>(1, 2, 3);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::OrbbecEgoFrameMetadataRecord::Pack(builder, record.get()));
    const auto* restored = flatbuffers::GetRoot<core::OrbbecEgoFrameMetadataRecord>(builder.GetBufferPointer());
    REQUIRE(restored->timestamp() != nullptr);
    CHECK(restored->timestamp()->available_time_local_common_clock() == 1);
    CHECK(restored->timestamp()->sample_time_local_common_clock() == 2);
    CHECK(restored->timestamp()->sample_time_raw_device_clock() == 3);
}

TEST_CASE("EGO embedded media schemas round trip", "[unit][ego][schema]")
{
    core::OrbbecEgoEncodedVideoFrameT video;
    video.stream = core::OrbbecEgoCameraStream_ColorRight;
    video.sequence_number = 42;
    video.width = 1600;
    video.height = 1300;
    video.fps = 30;
    video.pixel_format = core::OrbbecEgoPixelFormat_H264;
    video.encoded_data = { 0, 0, 0, 1, 0x65, 1 };
    video.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder video_builder;
    video_builder.Finish(core::OrbbecEgoEncodedVideoFrame::Pack(video_builder, &video));
    core::OrbbecEgoEncodedVideoFrameT restored_video;
    flatbuffers::GetRoot<core::OrbbecEgoEncodedVideoFrame>(video_builder.GetBufferPointer())->UnPackTo(&restored_video);
    CHECK(restored_video.stream == core::OrbbecEgoCameraStream_ColorRight);
    CHECK(restored_video.encoded_data == video.encoded_data);
    CHECK(restored_video.capture_epoch == 2);

    core::OrbbecEgoPcmAudioChunkT audio;
    audio.sequence_number = 7;
    audio.sample_rate_hz = 48000;
    audio.channel_count = 1;
    audio.bits_per_sample = 16;
    audio.sample_count = 2;
    audio.pcm_data = { 1, 0, 2, 0 };
    audio.capture_epoch = 2;
    flatbuffers::FlatBufferBuilder audio_builder;
    audio_builder.Finish(core::OrbbecEgoPcmAudioChunk::Pack(audio_builder, &audio));
    core::OrbbecEgoPcmAudioChunkT restored_audio;
    flatbuffers::GetRoot<core::OrbbecEgoPcmAudioChunk>(audio_builder.GetBufferPointer())->UnPackTo(&restored_audio);
    CHECK(restored_audio.pcm_data == audio.pcm_data);
    CHECK(restored_audio.capture_epoch == 2);
}

TEST_CASE("EGO diagnostic events preserve typed payloads and optional zero", "[unit][ego][schema]")
{
    core::OrbbecEgoDeviceStateT native;
    native.diagnostic_version = core::EGO_DIAGNOSTIC_VERSION;
    native.device_uid = "test-device";
    native.capture_epoch = 2;
    native.observed_at_ns = 100;
    SECTION("SDK report keeps a real zero error clear")
    {
        core::OrbbecEgoSdkReportT report;
        report.sequence_number = UINT64_MAX;
        report.error_flags = 0;
        native.event.Set(report);
        const auto state = core::pack<core::OrbbecEgoDeviceState>(native);
        CHECK_NOTHROW(core::validate_ego_diagnostic(*state));
        CHECK(state->event_as_OrbbecEgoSdkReport()->sequence_number() == UINT64_MAX);
        CHECK(state->event_as_OrbbecEgoSdkReport()->error_flags() == 0);
        CHECK(state->event_as_OrbbecEgoRecoveryEvent() == nullptr);
    }
    SECTION("final summary keeps cumulative losses")
    {
        core::OrbbecEgoCaptureSummaryT summary;
        summary.capture_health = core::OrbbecEgoCaptureHealth_Incomplete;
        summary.reason = "dropped video";
        summary.queue_capacity = 4096;
        summary.queue_peak = 99;
        summary.dropped_video_frame_sets = 3;
        summary.is_final = true;
        native.event.Set(summary);
        auto state = core::pack<core::OrbbecEgoDeviceState>(native);
        CHECK_NOTHROW(core::validate_ego_diagnostic(*state));
        auto retained = state.narrow(state->event_as_OrbbecEgoCaptureSummary());
        state.reset();
        CHECK(retained->dropped_video_frame_sets() == 3);
        CHECK(retained->is_final());
        CHECK(retained->reason()->str() == "dropped video");
    }
    SECTION("recovery carries only its transition")
    {
        core::OrbbecEgoRecoveryEventT recovery;
        recovery.stage = core::OrbbecEgoRecoveryStage_Recovered;
        recovery.reconnect_attempt = 4;
        native.event.Set(recovery);
        const auto state = core::pack<core::OrbbecEgoDeviceState>(native);
        CHECK_NOTHROW(core::validate_ego_diagnostic(*state));
        CHECK(state->capture_epoch() == 2);
        CHECK(state->event_as_OrbbecEgoRecoveryEvent()->reconnect_attempt() == 4);
    }
    SECTION("unavailable and zero temperature differ on the wire")
    {
        core::OrbbecEgoDeviceSnapshotT snapshot;
        snapshot.properties.emplace_back(279, 0);
        native.event.Set(snapshot);
        const auto missing = core::pack<core::OrbbecEgoDeviceState>(native);
        CHECK_NOTHROW(core::validate_ego_diagnostic(*missing));
        CHECK_FALSE(missing->event_as_OrbbecEgoDeviceSnapshot()->imu_temperature_c().has_value());
        snapshot.imu_temperature_c = 0.0f;
        native.event.Set(snapshot);
        const auto measured = core::pack<core::OrbbecEgoDeviceState>(native);
        const auto temperature = measured->event_as_OrbbecEgoDeviceSnapshot()->imu_temperature_c();
        REQUIRE(temperature.has_value());
        CHECK(*temperature == 0.0f);
        CHECK(measured->event_as_OrbbecEgoDeviceSnapshot()->properties()->Get(0)->value() == 0);
    }
}

TEST_CASE("EGO diagnostics reject missing events and unsupported versions", "[unit][ego][schema]")
{
    core::OrbbecEgoDeviceStateT native;
    native.device_uid = "test-device";
    native.observed_at_ns = 100;
    native.diagnostic_version = core::EGO_DIAGNOSTIC_VERSION;
    CHECK_THROWS_AS(
        core::validate_ego_diagnostic(*core::pack<core::OrbbecEgoDeviceState>(native)), std::invalid_argument);
    native.event.Set(core::OrbbecEgoSdkReportT{});
    native.diagnostic_version = 1;
    CHECK_THROWS_AS(
        core::validate_ego_diagnostic(*core::pack<core::OrbbecEgoDeviceState>(native)), std::invalid_argument);
    native.diagnostic_version = core::EGO_DIAGNOSTIC_VERSION;
    auto state = core::pack<core::OrbbecEgoDeviceState>(native);
    auto bytes = std::vector<uint8_t>(state.buffer().begin(), state.buffer().end());
    auto* mutable_state = flatbuffers::GetMutableRoot<core::OrbbecEgoDeviceState>(bytes.data());
    // The generated verifier accepts unknown union tags; the EGO consumer rejects them.
    auto* table = reinterpret_cast<flatbuffers::Table*>(mutable_state);
    REQUIRE(table->SetField<uint8_t>(core::OrbbecEgoDeviceState::VT_EVENT_TYPE, 99));
    CHECK_THROWS_AS(core::validate_ego_diagnostic(*mutable_state), std::invalid_argument);
}

TEST_CASE("EGO event protocol intentionally rejects the former diagnostic layout", "[unit][ego][schema]")
{
    flatbuffers::Parser old;
    REQUIRE(old.Parse(R"(
        namespace core;
        table OrbbecEgoDeviceState { sequence_number: uint64 (id: 0); device_uid: string (id: 1); }
        table OrbbecEgoDeviceStateRecord { data: OrbbecEgoDeviceState (id: 0); }
        root_type OrbbecEgoDeviceStateRecord;
    )"));
    old.Serialize();
    const auto current = core::OrbbecEgoDeviceStateRecordBinarySchema::data();
    const auto result = core::check_schema_compat({ old.builder_.GetBufferPointer(), old.builder_.GetSize() },
                                                  { current, core::OrbbecEgoDeviceStateRecordBinarySchema::size() });
    CHECK(result.status == core::SchemaCompat::Incompatible);
}
