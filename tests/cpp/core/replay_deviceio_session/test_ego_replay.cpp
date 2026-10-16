// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/replay_session.hpp>
#include <deviceio_trackers/ego_audio_tracker.hpp>
#include <deviceio_trackers/ego_calibration_tracker.hpp>
#include <deviceio_trackers/ego_device_state_tracker.hpp>
#include <deviceio_trackers/ego_frame_metadata_tracker.hpp>
#include <deviceio_trackers/ego_imu_tracker.hpp>
#include <mcap/recording_traits.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/ego_audio_bfbs_generated.h>
#include <schema/ego_calibration_bfbs_generated.h>
#include <schema/ego_camera_bfbs_generated.h>
#include <schema/ego_device_state_bfbs_generated.h>
#include <schema/ego_imu_bfbs_generated.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace
{

struct Recording
{
    std::string path;
    Recording()
    {
        static std::atomic<unsigned> counter{ 0 };
        const auto time = std::chrono::steady_clock::now().time_since_epoch().count();
        path = (std::filesystem::temp_directory_path() /
                ("ego_replay_" + std::to_string(time) + "_" + std::to_string(counter++) + ".mcap"))
                   .string();
    }
    ~Recording()
    {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

std::unique_ptr<mcap::McapReader> open_reader(const std::string& path)
{
    auto reader = std::make_unique<mcap::McapReader>();
    REQUIRE(reader->open(path).ok());
    return reader;
}

std::vector<std::string> channels(auto values)
{
    return { values.begin(), values.end() };
}

} // namespace

TEST_CASE("EGO replay reads all seven raw streams and retains snapshots", "[unit][replay][ego]")
{
    bool include_tracked = false;
    SECTION("legacy recording has only raw topics")
    {
    }
    SECTION("new recording also has tracked auxiliary topics")
    {
        include_tracked = true;
    }

    Recording recording;
    {
        mcap::McapWriter writer;
        mcap::McapWriterOptions options("ego-test");
        options.compression = mcap::Compression::None;
        REQUIRE(writer.open(recording.path, options).ok());
        core::McapTrackerChannels<core::EgoFrameMetadataRecord> camera(
            writer, "ego_metadata", { "ColorLeft", "ColorRight" });
        core::McapTrackerChannels<core::EgoImuBatchRecord> imu(writer, "ego_imu", { "Accel", "Gyro" });
        core::McapTrackerChannels<core::EgoAudioChunkRecord> audio(
            writer, "ego_audio",
            include_tracked ? channels(core::EgoAudioRecordingTraits::recording_channels) :
                              std::vector<std::string>{ "Audio" });
        core::McapTrackerChannels<core::EgoCalibrationRecord> calibration(
            writer, "ego_calibration",
            include_tracked ? channels(core::EgoCalibrationRecordingTraits::recording_channels) :
                              std::vector<std::string>{ "Calibration" });
        core::McapTrackerChannels<core::EgoDeviceStateRecord> state(
            writer, "ego_device",
            include_tracked ? channels(core::EgoDeviceStateRecordingTraits::recording_channels) :
                              std::vector<std::string>{ "DeviceState" });
        core::McapTrackerChannels<core::EgoEncodedVideoFrameRecord> media(writer, "ego_media", { "ColorLeft" });
        for (uint64_t sequence = 1; sequence <= 3; ++sequence)
        {
            const core::DeviceDataTimestamp timestamp(sequence * 100, sequence * 90, sequence * 80);
            const bool inactive = sequence == 2;
            core::EgoFrameMetadataT frame;
            frame.sequence_number = sequence;
            frame.capture_epoch = 3;
            frame.stream = core::EgoCameraStream_ColorLeft;
            camera.write(0, core::pack_record<core::EgoFrameMetadataRecord>(inactive ? nullptr : &frame, timestamp));
            frame.stream = core::EgoCameraStream_ColorRight;
            camera.write(1, core::pack_record<core::EgoFrameMetadataRecord>(inactive ? nullptr : &frame, timestamp));
            core::EgoImuBatchT batch;
            batch.sequence_number = sequence;
            batch.sensor = core::EgoImuSensor_Accel;
            batch.samples.emplace_back(1, 2, 3, 24, 90, 80);
            imu.write(0, core::pack_record<core::EgoImuBatchRecord>(inactive ? nullptr : &batch, timestamp));
            batch.sensor = core::EgoImuSensor_Gyro;
            imu.write(1, core::pack_record<core::EgoImuBatchRecord>(inactive ? nullptr : &batch, timestamp));
            core::EgoAudioChunkT chunk;
            chunk.sequence_number = sequence;
            chunk.wav_data_offset = 44;
            const auto audio_record =
                core::pack_record<core::EgoAudioChunkRecord>(inactive ? nullptr : &chunk, timestamp);
            audio.write(0, audio_record);
            core::EgoCalibrationT calib;
            calib.device_uid = "ego_" + std::to_string(sequence);
            const auto calibration_record =
                core::pack_record<core::EgoCalibrationRecord>(inactive ? nullptr : &calib, timestamp);
            calibration.write(0, calibration_record);
            core::EgoDeviceStateT device;
            device.device_uid = "ego";
            device.capture_epoch = 3;
            device.observed_at_ns = sequence * 90;
            device.diagnostic_version = 2;
            core::EgoSdkReportT report;
            report.sequence_number = sequence;
            device.event.Set(report);
            const auto state_record =
                core::pack_record<core::EgoDeviceStateRecord>(inactive ? nullptr : &device, timestamp);
            state.write(0, state_record);
            if (include_tracked)
            {
                audio.write(1, audio_record);
                calibration.write(1, calibration_record);
                state.write(1, state_record);
            }
            core::EgoEncodedVideoFrameT video;
            video.encoded_data = { 0, 0, 0, 1, 0x65 };
            media.write(0, core::pack_record<core::EgoEncodedVideoFrameRecord>(&video, timestamp));
        }
        writer.close();
    }

    core::EgoFrameMetadataTracker camera(
        "ego_metadata", { core::EgoCameraStream_ColorLeft, core::EgoCameraStream_ColorRight });
    core::EgoImuTracker imu("ego_imu");
    core::EgoAudioTracker audio("ego_audio/Audio");
    core::EgoCalibrationTracker calibration("ego_calibration/Calibration");
    core::EgoDeviceStateTracker state("ego_device/DeviceState");
    auto session = core::ReplaySession::run({ recording.path,
                                              {
                                                  { &camera, "ego_metadata" },
                                                  { &imu, "ego_imu" },
                                                  { &audio, "ego_audio" },
                                                  { &calibration, "ego_calibration" },
                                                  { &state, "ego_device" },
                                              } });
    CHECK_FALSE(camera.get_stream_data(*session, 0));
    CHECK_FALSE(audio.get_data(*session));
    session->update();
    const auto snapshot = camera.get_stream_data(*session, 0);
    REQUIRE(snapshot);
    for (uint64_t sequence = 1; sequence <= 3; ++sequence)
    {
        if (sequence > 1)
            session->update();
        if (sequence == 2)
        {
            CHECK_FALSE(camera.get_stream_data(*session, 0));
            CHECK_FALSE(camera.get_stream_data(*session, 1));
            CHECK_FALSE(imu.get_stream_data(*session, 0));
            CHECK_FALSE(imu.get_stream_data(*session, 1));
            CHECK_FALSE(audio.get_data(*session));
            CHECK_FALSE(calibration.get_data(*session));
            CHECK_FALSE(state.get_data(*session));
            continue;
        }
        REQUIRE(camera.get_stream_data(*session, 1));
        CHECK(camera.get_stream_data(*session, 0)->sequence_number() == sequence);
        CHECK(camera.get_stream_data(*session, 1)->stream() == core::EgoCameraStream_ColorRight);
        REQUIRE(imu.get_stream_data(*session, 0));
        REQUIRE(imu.get_stream_data(*session, 1));
        CHECK(imu.get_stream_data(*session, 0)->sensor() == core::EgoImuSensor_Accel);
        CHECK(imu.get_stream_data(*session, 1)->sensor() == core::EgoImuSensor_Gyro);
        REQUIRE(audio.get_data(*session));
        CHECK(audio.get_data(*session)->sequence_number() == sequence);
        REQUIRE(calibration.get_data(*session));
        CHECK(calibration.get_data(*session)->device_uid()->str() == "ego_" + std::to_string(sequence));
        REQUIRE(state.get_data(*session));
        CHECK(state.get_data(*session)->event_as_EgoSdkReport()->sequence_number() == sequence);
    }
    session->update();
    CHECK_FALSE(camera.get_stream_data(*session, 0));
    CHECK_FALSE(camera.get_stream_data(*session, 1));
    CHECK_FALSE(imu.get_stream_data(*session, 0));
    CHECK_FALSE(imu.get_stream_data(*session, 1));
    CHECK_FALSE(audio.get_data(*session));
    CHECK_FALSE(calibration.get_data(*session));
    CHECK_FALSE(state.get_data(*session));
    CHECK_THROWS_AS(camera.get_stream_data(*session, 2), std::out_of_range);
    CHECK_THROWS_AS(imu.get_stream_data(*session, 2), std::out_of_range);
    session.reset();
    CHECK(snapshot->sequence_number() == 1);
    CHECK(snapshot->capture_epoch() == 3);
}

TEST_CASE("EGO secondary media records reuse the unchanged file schemas", "[unit][replay][ego][mcap]")
{
    Recording recording;
    {
        mcap::McapWriter writer;
        mcap::McapWriterOptions options("ego-test");
        options.compression = mcap::Compression::None;
        REQUIRE(writer.open(recording.path, options).ok());
        core::McapTrackerChannels<core::EgoEncodedVideoFrameRecord> video(writer, "ego_media", { "ColorLeft" });
        core::McapTrackerChannels<core::EgoPcmAudioChunkRecord> audio(writer, "ego_media", { "Audio" });
        core::EgoEncodedVideoFrameT frame;
        frame.encoded_data = { 0, 0, 0, 1, 0x65 };
        video.write(
            0, core::pack_record<core::EgoEncodedVideoFrameRecord>(&frame, core::DeviceDataTimestamp(100, 90, 80)));
        core::EgoPcmAudioChunkT chunk;
        chunk.pcm_data = { 1, 0, 2, 0 };
        audio.write(0, core::pack_record<core::EgoPcmAudioChunkRecord>(&chunk, core::DeviceDataTimestamp(100, 90, 80)));
        writer.close();
    }
    auto inventory_reader = open_reader(recording.path);
    const core::RecordedSchemas recorded(*inventory_reader);
    core::McapTrackerViewers<core::EgoEncodedVideoFrameRecord> video(
        open_reader(recording.path), "ego_media", { "ColorLeft" }, recorded);
    core::McapTrackerViewers<core::EgoPcmAudioChunkRecord> audio(
        open_reader(recording.path), "ego_media", { "Audio" }, recorded);
    const auto frame = video.read(0);
    REQUIRE(frame);
    CHECK(frame->data()->encoded_data()->Get(4) == 0x65);
    CHECK(frame->timestamp()->sample_time_raw_device_clock() == 80);
    const auto chunk = audio.read(0);
    REQUIRE(chunk);
    CHECK(chunk->data()->pcm_data()->Get(2) == 2);
    CHECK_FALSE(video.read(0));
    CHECK_FALSE(audio.read(0));
}
