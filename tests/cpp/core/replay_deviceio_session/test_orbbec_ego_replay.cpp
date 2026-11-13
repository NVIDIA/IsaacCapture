// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/replay_session.hpp>
#include <deviceio_trackers/orbbec_ego_calibration_tracker.hpp>
#include <deviceio_trackers/orbbec_ego_device_state_tracker.hpp>
#include <deviceio_trackers/orbbec_ego_frame_metadata_tracker.hpp>
#include <deviceio_trackers/orbbec_ego_imu_tracker.hpp>
#include <mcap/recording_traits.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/orbbec_ego_calibration_bfbs_generated.h>
#include <schema/orbbec_ego_camera_bfbs_generated.h>
#include <schema/orbbec_ego_device_state_bfbs_generated.h>
#include <schema/orbbec_ego_imu_bfbs_generated.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
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

std::vector<std::string> channels(auto values)
{
    return { values.begin(), values.end() };
}

} // namespace

TEST_CASE("EGO replay reads every raw stream and retains snapshots", "[unit][replay][ego]")
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
        const auto frame_channels = include_tracked ?
                                        channels(core::OrbbecEgoFrameMetadataRecordingTraits::recording_channels) :
                                        std::vector<std::string>{ "FrameMetadata" };
        core::McapTrackerChannels<core::OrbbecEgoFrameMetadataRecord> left(
            writer, "orbbec_ego_metadata_colorleft", frame_channels);
        core::McapTrackerChannels<core::OrbbecEgoFrameMetadataRecord> right(
            writer, "orbbec_ego_metadata_colorright", frame_channels);
        const auto imu_channels = include_tracked ? channels(core::OrbbecEgoImuRecordingTraits::recording_channels) :
                                                    std::vector<std::string>{ "Imu" };
        core::McapTrackerChannels<core::OrbbecEgoImuBatchRecord> accel(writer, "orbbec_ego_imu_accel", imu_channels);
        core::McapTrackerChannels<core::OrbbecEgoImuBatchRecord> gyro(writer, "orbbec_ego_imu_gyro", imu_channels);
        core::McapTrackerChannels<core::OrbbecEgoCalibrationRecord> calibration(
            writer, "orbbec_ego_calibration",
            include_tracked ? channels(core::OrbbecEgoCalibrationRecordingTraits::recording_channels) :
                              std::vector<std::string>{ "Calibration" });
        core::McapTrackerChannels<core::OrbbecEgoDeviceStateRecord> state(
            writer, "orbbec_ego_device",
            include_tracked ? channels(core::OrbbecEgoDeviceStateRecordingTraits::recording_channels) :
                              std::vector<std::string>{ "DeviceState" });
        for (uint64_t sequence = 1; sequence <= 3; ++sequence)
        {
            const core::DeviceDataTimestamp timestamp(sequence * 100, sequence * 90, sequence * 80);
            const bool inactive = sequence == 2;
            core::OrbbecEgoFrameMetadataT frame;
            frame.sequence_number = sequence;
            frame.capture_epoch = 3;
            frame.stream = core::OrbbecEgoCameraStream_ColorLeft;
            const auto left_record =
                core::pack_record<core::OrbbecEgoFrameMetadataRecord>(inactive ? nullptr : &frame, timestamp);
            left.write(0, left_record);
            frame.stream = core::OrbbecEgoCameraStream_ColorRight;
            const auto right_record =
                core::pack_record<core::OrbbecEgoFrameMetadataRecord>(inactive ? nullptr : &frame, timestamp);
            right.write(0, right_record);
            core::OrbbecEgoImuBatchT batch;
            batch.sequence_number = sequence;
            batch.sensor = core::OrbbecEgoImuSensor_Accel;
            batch.samples.emplace_back(1, 2, 3, 90, 80);
            const auto accel_record =
                core::pack_record<core::OrbbecEgoImuBatchRecord>(inactive ? nullptr : &batch, timestamp);
            accel.write(0, accel_record);
            batch.sensor = core::OrbbecEgoImuSensor_Gyro;
            const auto gyro_record =
                core::pack_record<core::OrbbecEgoImuBatchRecord>(inactive ? nullptr : &batch, timestamp);
            gyro.write(0, gyro_record);
            core::OrbbecEgoCalibrationT calib;
            calib.device_uid = "ego_" + std::to_string(sequence);
            const auto calibration_record =
                core::pack_record<core::OrbbecEgoCalibrationRecord>(inactive ? nullptr : &calib, timestamp);
            calibration.write(0, calibration_record);
            core::OrbbecEgoDeviceStateT device;
            device.device_uid = "ego";
            device.capture_epoch = 3;
            device.observed_at_ns = sequence * 90;
            core::OrbbecEgoSdkReportT report;
            report.sequence_number = sequence;
            device.event.Set(report);
            const auto state_record =
                core::pack_record<core::OrbbecEgoDeviceStateRecord>(inactive ? nullptr : &device, timestamp);
            state.write(0, state_record);
            if (include_tracked)
            {
                left.write(1, left_record);
                right.write(1, right_record);
                accel.write(1, accel_record);
                gyro.write(1, gyro_record);
                calibration.write(1, calibration_record);
                state.write(1, state_record);
            }
        }
        writer.close();
    }

    core::OrbbecEgoFrameMetadataTracker left("ego/ColorLeft");
    core::OrbbecEgoFrameMetadataTracker right("ego/ColorRight");
    core::OrbbecEgoImuTracker accel("ego/Accel");
    core::OrbbecEgoImuTracker gyro("ego/Gyro");
    core::OrbbecEgoCalibrationTracker calibration("ego_calibration/Calibration");
    core::OrbbecEgoDeviceStateTracker state("ego_device/DeviceState");
    auto session = core::ReplaySession::run({ recording.path,
                                              {
                                                  { &left, "orbbec_ego_metadata_colorleft" },
                                                  { &right, "orbbec_ego_metadata_colorright" },
                                                  { &accel, "orbbec_ego_imu_accel" },
                                                  { &gyro, "orbbec_ego_imu_gyro" },
                                                  { &calibration, "orbbec_ego_calibration" },
                                                  { &state, "orbbec_ego_device" },
                                              } });
    CHECK_FALSE(left.get_data(*session));
    session->update();
    const auto snapshot = left.get_data(*session);
    REQUIRE(snapshot);
    for (uint64_t sequence = 1; sequence <= 3; ++sequence)
    {
        if (sequence > 1)
            session->update();
        if (sequence == 2)
        {
            CHECK_FALSE(left.get_data(*session));
            CHECK_FALSE(right.get_data(*session));
            CHECK_FALSE(accel.get_data(*session));
            CHECK_FALSE(gyro.get_data(*session));
            CHECK_FALSE(calibration.get_data(*session));
            CHECK_FALSE(state.get_data(*session));
            continue;
        }
        REQUIRE(right.get_data(*session));
        CHECK(left.get_data(*session)->sequence_number() == sequence);
        CHECK(right.get_data(*session)->stream() == core::OrbbecEgoCameraStream_ColorRight);
        REQUIRE(accel.get_data(*session));
        REQUIRE(gyro.get_data(*session));
        CHECK(accel.get_data(*session)->sensor() == core::OrbbecEgoImuSensor_Accel);
        CHECK(gyro.get_data(*session)->sensor() == core::OrbbecEgoImuSensor_Gyro);
        REQUIRE(calibration.get_data(*session));
        CHECK(calibration.get_data(*session)->device_uid()->str() == "ego_" + std::to_string(sequence));
        REQUIRE(state.get_data(*session));
        CHECK(state.get_data(*session)->event_as_OrbbecEgoSdkReport()->sequence_number() == sequence);
    }
    session->update();
    CHECK_FALSE(left.get_data(*session));
    CHECK_FALSE(right.get_data(*session));
    CHECK_FALSE(accel.get_data(*session));
    CHECK_FALSE(gyro.get_data(*session));
    CHECK_FALSE(calibration.get_data(*session));
    CHECK_FALSE(state.get_data(*session));
    session.reset();
    CHECK(snapshot->sequence_number() == 1);
    CHECK(snapshot->capture_epoch() == 3);
}
