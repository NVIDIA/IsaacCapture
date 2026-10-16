// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/ego_audio_tracker.hpp>
#include <deviceio_trackers/ego_calibration_tracker.hpp>
#include <deviceio_trackers/ego_device_state_tracker.hpp>
#include <deviceio_trackers/ego_frame_metadata_tracker.hpp>
#include <deviceio_trackers/ego_imu_tracker.hpp>
#include <live_trackers/schema_tracker_base.hpp>

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("EGO live dispatch resolves only tensor and time extensions", "[unit][live_trackers][ego]")
{
    std::vector<std::shared_ptr<core::ITracker>> trackers = {
        std::make_shared<core::EgoFrameMetadataTracker>(
            "ego_metadata",
            std::vector<core::EgoCameraStream>{ core::EgoCameraStream_ColorLeft, core::EgoCameraStream_ColorRight }),
        std::make_shared<core::EgoImuTracker>("ego_imu"),
        std::make_shared<core::EgoAudioTracker>("ego_audio/Audio"),
        std::make_shared<core::EgoCalibrationTracker>("ego_calibration/Calibration"),
        std::make_shared<core::EgoDeviceStateTracker>("ego_device/DeviceState"),
    };
    const auto expected = core::SchemaTrackerBase::get_required_extensions();
    const std::set<std::string> expected_set(expected.begin(), expected.end());
    for (const auto& tracker : trackers)
    {
        const auto extensions = core::DeviceIOSession::get_required_extensions({ tracker });
        CHECK(std::set<std::string>(extensions.begin(), extensions.end()) == expected_set);
    }
    const auto extensions = core::DeviceIOSession::get_required_extensions(trackers);
    CHECK(std::set<std::string>(extensions.begin(), extensions.end()) == expected_set);
}

TEST_CASE("EGO multi-stream trackers reject ambiguous collection routing", "[unit][live_trackers][ego]")
{
    using Camera = core::EgoFrameMetadataTracker;
    using Imu = core::EgoImuTracker;
    CHECK_THROWS_AS(Camera("", { core::EgoCameraStream_ColorLeft }), std::invalid_argument);
    CHECK_THROWS_AS(Camera("camera", {}), std::invalid_argument);
    CHECK_THROWS_AS(
        Camera("camera", { core::EgoCameraStream_ColorLeft, core::EgoCameraStream_ColorLeft }), std::invalid_argument);
    CHECK_THROWS_AS(Camera("camera", { static_cast<core::EgoCameraStream>(99) }), std::invalid_argument);
    CHECK_THROWS_AS(Imu("imu", { core::EgoImuSensor_Accel, core::EgoImuSensor_Accel }), std::invalid_argument);
    CHECK_THROWS_AS(Imu("imu", { static_cast<core::EgoImuSensor>(99) }), std::invalid_argument);
}
