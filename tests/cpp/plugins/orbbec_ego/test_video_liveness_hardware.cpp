// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <ego_camera/ego_camera.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

class DelayedMetadataSink final : public plugins::ego::IMetadataSink
{
public:
    explicit DelayedMetadataSink(std::chrono::seconds delay) : delay_(delay)
    {
    }

    void on_frame_metadata(const plugins::ego::CapturedFrame& frame) override
    {
        ++frames[frame.metadata.stream];
        if (!delayed_)
        {
            delayed_ = true;
            std::this_thread::sleep_for(delay_);
        }
    }

    void on_capture_status(const plugins::ego::CaptureStatus& value) override
    {
        status = value;
    }

    void commit() override
    {
        committed = true;
    }

    std::map<core::OrbbecEgoCameraStream, uint64_t> frames;
    plugins::ego::CaptureStatus status;
    bool committed = false;

private:
    std::chrono::seconds delay_;
    bool delayed_ = false;
};

void check_capture_liveness(std::chrono::seconds sink_delay, std::chrono::seconds consumer_pause)
{
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("isaaccapture-ego-liveness-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    INFO("Capture evidence: " << directory.string());
    REQUIRE(std::filesystem::create_directory(directory));

    const std::vector<plugins::ego::StreamConfig> streams{
        { core::OrbbecEgoCameraStream_ColorLeft, (directory / "ColorLeft.h264").string(),
          core::OrbbecEgoPixelFormat_H264, 1600, 1300, 30 },
        { core::OrbbecEgoCameraStream_ColorRight, (directory / "ColorRight.h264").string(),
          core::OrbbecEgoPixelFormat_H264, 1600, 1300, 30 }
    };
    plugins::ego::CaptureConfig config;
    config.reconnect_timeout_seconds = 0;
    config.enable_imu = false;
    auto metadata = std::make_unique<DelayedMetadataSink>(sink_delay);
    auto* observed = metadata.get();
    plugins::ego::EgoCamera camera(
        config, streams, std::make_unique<plugins::ego::FrameSink>(streams, std::move(metadata)));

    std::this_thread::sleep_for(consumer_pause);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool update_succeeded = false;
    do
    {
        update_succeeded = false;
        CHECK_NOTHROW(
            [&]
            {
                camera.update();
                update_succeeded = true;
            }());
    } while (update_succeeded && observed->frames.empty() && std::chrono::steady_clock::now() < deadline);

    INFO("Capture status: " << observed->status.error);
    CHECK_FALSE(observed->frames.empty());
    CHECK(observed->status.availability == plugins::ego::CaptureAvailability::Streaming);
    CHECK_NOTHROW(camera.close());
    CHECK(observed->committed);
    CHECK(camera.auxiliary_stats().reconnect_attempts == 0);
    // Read callback-owned counters only after close has stopped the producers.
    const auto dropped = camera.auxiliary_stats().dropped_video_frame_sets;
    if (consumer_pause > std::chrono::seconds::zero())
        CHECK(dropped > 0);
    for (const auto& stream : streams)
    {
        CHECK(observed->frames[stream.camera] > 0);
        CHECK(std::filesystem::file_size(stream.output_path) > 0);
    }

    if (update_succeeded && observed->committed && (consumer_pause == std::chrono::seconds::zero() || dropped > 0))
    {
        std::filesystem::remove_all(directory);
    }
}

} // namespace

TEST_CASE("EGO video remains live while its metadata consumer blocks", "[.hardware][ego][liveness]")
{
    check_capture_liveness(std::chrono::seconds(6), std::chrono::seconds(0));
}

TEST_CASE("EGO video remains live while its full queue drops incoming frames", "[.hardware][ego][liveness]")
{
    // At 30 FPS, this fills the queue and ages its newest retained frame past the liveness deadline.
    check_capture_liveness(std::chrono::seconds(0), std::chrono::seconds(15));
}
