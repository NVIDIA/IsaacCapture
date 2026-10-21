// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <remote_plugin_session/latency_stats.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace remote_plugin_session;

namespace
{

constexpr int64_t kMs = 1'000'000;

std::string read_file(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

} // namespace

TEST_CASE("index_curl_m is the index-tip to wrist distance", "[remote_device_bridge][unit]")
{
    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
    joints[XR_HAND_JOINT_WRIST_EXT].pose.position = { 1.0f, 2.0f, 3.0f };
    joints[XR_HAND_JOINT_INDEX_TIP_EXT].pose.position = { 1.03f, 2.04f, 3.0f };
    joints[XR_HAND_JOINT_PALM_EXT].pose.position = { 9.0f, 9.0f, 9.0f }; // ignored
    REQUIRE(index_curl_m(joints.data()) == Catch::Approx(0.05f).epsilon(1e-4));
}

TEST_CASE("format_clock_stats renders the 5 s line suffix", "[remote_device_bridge][unit]")
{
    ClockOffsetEstimator clock;
    std::vector<int64_t> ages;
    REQUIRE(format_clock_stats(ages, clock) ==
            " | age_ms p50=- p99=- max=- | rtt_ms min=- last=- | offset_ms=0.000 clock=one-way");

    // Offset 5 s, 70 ms each way, then a slower 151 ms exchange.
    clock.add_two_way_sample(0, 70 * kMs - 5'000 * kMs, 71 * kMs - 5'000 * kMs, 141 * kMs);
    clock.add_two_way_sample(1'000 * kMs, 1'080 * kMs - 5'000 * kMs, 1'081 * kMs - 5'000 * kMs, 1'152 * kMs);
    for (int i = 1; i <= 100; ++i)
    {
        ages.push_back(i * kMs / 2); // 0.5 .. 50 ms
    }
    const std::string line = format_clock_stats(ages, clock);
    INFO(line);
    REQUIRE(line ==
            " | age_ms p50=25.0 p99=49.5 max=50.0 | rtt_ms min=140.0 last=151.0 | offset_ms=5000.000 clock=two-way");
}

TEST_CASE("SampleCsv writes the header once and one row per sample", "[remote_device_bridge][unit]")
{
    const SampleCsvRow row{ 1'000'000'123, 1'000'000'100, 42'000, 999'930'000, 70'123'456, "right", 17, 0.08765f };
    char line[256];
    REQUIRE(SampleCsv::format(line, sizeof(line), row) > 0);
    REQUIRE(std::string(line) == "1000000123,1000000100,42000,999930000,70.123,right,17,0.08765\n");

    const std::string path =
        (std::filesystem::temp_directory_path() / ("remote_plugin_session_csv_" + std::to_string(::getpid()) + ".csv"))
            .string();
    SampleCsv::create(path);
    {
        SampleCsv first(path);
        first.write(row);
    }
    {
        SampleCsv second(path); // a later client appends
        second.write(row);
    }
    REQUIRE(read_file(path) == std::string(SampleCsv::kHeader) + line + line);
    REQUIRE(std::string(SampleCsv::kHeader) ==
            "push_mono_ns,recv_mono_ns,capture_sender_ns,capture_mapped_ns,age_ms,side,seq,curl\n");
    std::remove(path.c_str());

    REQUIRE_THROWS_AS(SampleCsv("/nonexistent_dir/x.csv"), std::runtime_error);
}
