// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SensingCamera end to end on the mock backend: capture, convert, encode, sink.

#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <core/frame_sink.hpp>
#include <core/sensing_camera.hpp>

#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>

using namespace plugins::sensing;
using namespace sensing_test;
using namespace std::chrono_literals;

namespace
{

struct Recorded
{
    uint32_t sensor_id;
    uint64_t sequence_number;
    uint64_t capture_tsc_ns;
    int64_t host_ns;
};

/// Collects what would have gone to MCAP.
struct Recorder
{
    std::mutex mutex;
    std::vector<Recorded> records;
};

class RecordingPusher : public IMetadataPusher
{
public:
    explicit RecordingPusher(std::shared_ptr<Recorder> recorder) : m_recorder(std::move(recorder))
    {
    }

    void on_frame_metadata(const core::FrameMetadataSensingT& metadata,
                           int64_t sample_time_local_common_clock_ns,
                           int64_t sample_time_raw_device_clock_ns) override
    {
        (void)sample_time_raw_device_clock_ns;
        std::lock_guard<std::mutex> lock(m_recorder->mutex);
        m_recorder->records.push_back({ metadata.sensor_id, metadata.sequence_number, metadata.capture_tsc_ns,
                                        sample_time_local_common_clock_ns });
    }

private:
    std::shared_ptr<Recorder> m_recorder;
};

SensingConfig make_config()
{
    SensingConfig config;
    config.platform_config_json = SENSING_PLATFORM_CONFIG;
    config.bitrate_bps = 2'000'000;
    return config;
}

std::vector<uint8_t> read_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
}

std::vector<Recorded> records_for(const Recorder& recorder, uint32_t sensor)
{
    std::vector<Recorded> out;
    for (const auto& r : recorder.records)
    {
        if (r.sensor_id == sensor)
            out.push_back(r);
    }
    return out;
}

// Runs the plugin's update loop for `duration`, then flushes.
void run_for(SensingCamera& camera, std::chrono::milliseconds duration)
{
    const auto end = std::chrono::steady_clock::now() + duration;
    while (std::chrono::steady_clock::now() < end)
    {
        camera.update();
        std::this_thread::sleep_for(2ms);
    }
    camera.flush();
}

} // namespace

TEST_CASE("an unknown sensor index is reported with the available pipelines", "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    std::vector<StreamConfig> streams(1);
    streams[0].sensor_id = 5;
    streams[0].ipc_socket_path = unique_path("sensing_mock", ".sock");
    CHECK_THROWS_WITH(SensingCamera(make_config(), streams, std::make_unique<FrameSink>(streams)),
                      Catch::Matchers::ContainsSubstring("not a pipeline"));
}

TEST_CASE("an encoded stream writes Annex-B data and one metadata record per unit", "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    const std::string output = unique_path("sensing_mock", ".h264");

    std::vector<StreamConfig> streams(1);
    streams[0].sensor_id = 0;
    streams[0].output_path = output;

    auto recorder = std::make_shared<Recorder>();
    {
        SensingCamera camera(
            make_config(), streams, std::make_unique<FrameSink>(streams, std::make_unique<RecordingPusher>(recorder)));
        run_for(camera, 1000ms);
    }

    const auto bytes = read_file(output);
    std::remove(output.c_str());
    REQUIRE(bytes.size() > 4);

    const auto nals = split_annex_b(bytes);
    REQUIRE_FALSE(nals.empty());
    CHECK(nals[0].type == 0x67); // the stream opens with its parameter sets
    std::vector<uint32_t> frames;
    for (const auto& nal : nals)
    {
        if (is_slice(nal))
            frames.push_back(frame_of(nal));
    }
    REQUIRE(frames.size() >= 10);
    // The encoder preserves submission order; the mailbox may skip frames but never reorders.
    for (size_t i = 1; i < frames.size(); ++i)
        CHECK(frames[i] > frames[i - 1]);

    const auto records = records_for(*recorder, 0);
    REQUIRE(records.size() >= 5);
    for (size_t i = 0; i < records.size(); ++i)
        CHECK(records[i].sequence_number == i); // contiguous from zero, one per dispatched unit
    // The final flush folds the warm-up tail into one unit, so units can be fewer than frames.
    CHECK(records.size() <= frames.size());
}

TEST_CASE("an ipc-only stream records metadata with the sensor's capture TSC", "[unit][gpu][sensing]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    std::vector<StreamConfig> streams(2);
    streams[0].sensor_id = 0;
    streams[0].ipc_socket_path = unique_path("sensing_mock", ".sock");
    streams[1].sensor_id = 1;
    streams[1].ipc_socket_path = unique_path("sensing_mock", ".sock");

    auto recorder = std::make_shared<Recorder>();
    {
        SensingCamera camera(
            make_config(), streams, std::make_unique<FrameSink>(streams, std::make_unique<RecordingPusher>(recorder)));
        run_for(camera, 700ms);
    }

    std::vector<Recorded> per_sensor[2] = { records_for(*recorder, 0), records_for(*recorder, 1) };
    for (uint32_t sensor = 0; sensor < 2; ++sensor)
    {
        INFO("sensor " << sensor);
        REQUIRE(per_sensor[sensor].size() >= 5);
        for (size_t i = 0; i < per_sensor[sensor].size(); ++i)
        {
            CHECK(per_sensor[sensor][i].sequence_number == i);
            CHECK(per_sensor[sensor][i].capture_tsc_ns != 0);
        }
    }

    // The same frame carries the same TSC on both sensors, so the two eyes can be paired on it.
    std::set<uint64_t> left;
    for (const auto& r : per_sensor[0])
        left.insert(r.capture_tsc_ns);
    int shared = 0;
    for (const auto& r : per_sensor[1])
        shared += static_cast<int>(left.count(r.capture_tsc_ns));
    CHECK(shared >= 3);
}

// The encoder returns each unit's stamp at microsecond resolution, while SensingCamera remembers the
// stamp it submitted in nanoseconds and matches them for equality, so the capture TSC of an encoded
// stream is never found. Remove [!shouldfail] when that is fixed.
TEST_CASE("an encoded stream keeps its capture TSC through the encoder", "[unit][gpu][sensing][!shouldfail]")
{
    ScopedEnv scale("SENSING_MOCK_SCALE_DIV", "8");
    const std::string output = unique_path("sensing_mock", ".h264");

    std::vector<StreamConfig> streams(1);
    streams[0].sensor_id = 0;
    streams[0].output_path = output;

    auto recorder = std::make_shared<Recorder>();
    {
        SensingCamera camera(
            make_config(), streams, std::make_unique<FrameSink>(streams, std::make_unique<RecordingPusher>(recorder)));
        run_for(camera, 700ms);
    }
    std::remove(output.c_str());

    const auto records = records_for(*recorder, 0);
    REQUIRE(records.size() >= 5);
    // The last record is the flushed tail, whose stamp is not matched either way.
    for (size_t i = 0; i + 1 < records.size(); ++i)
        CHECK(records[i].capture_tsc_ns != 0);
}
