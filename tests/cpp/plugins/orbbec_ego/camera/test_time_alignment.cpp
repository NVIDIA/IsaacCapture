// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <ego_camera/time_alignment.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
const std::string kProfile = R"({
"version":1,"scope":"capture_profile","status":"validated","model_id":"test-relative-only",
"max_capture_duration_ns":900000000000,
"reference_stream":"left_rgb","identity":{"serial":"S1","firmware":"1.1.1","sdk_version":"2.9.0",
"sdk_binary_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
"profile":{"video":{"format":"h264","width":1600,"height":1300,"fps":30},"imu_rate_hz":1000,
"audio":{"sample_rate_hz":48000,"channels":1,"bits_per_sample":16}},
"streams":{"left_rgb":{"offset_ns":0,"scale":1},"right_rgb":{"offset_ns":-1000,"scale":1},
"accel":{"offset_ns":-2000,"scale":1},"gyro":{"offset_ns":-3000,"scale":1}},
"validation":{"method":"synthetic unit fixture, not physical evidence","held_out_duration_ns":1800000000000,
"max_abs_error_ns":1000,"uncertainty_ns":1000,"evidence":["synthetic"]}})";

plugins::ego::TimingIdentity identity()
{
    return { "S1", "1.1.1", "2.9.0", "h264", 1600, 1300, 30, 1000, 48000, 1, 16, true, std::string(64, 'a') };
}

struct ProfileFile
{
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("ego_time_profile_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    explicit ProfileFile(const std::string& text = kProfile)
    {
        std::ofstream output(path);
        output << text;
    }
    ~ProfileFile()
    {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

std::string replace(std::string text, const std::string& before, const std::string& after)
{
    const auto position = text.find(before);
    REQUIRE(position != std::string::npos);
    text.replace(position, before.size(), after);
    return text;
}
} // namespace

TEST_CASE("EGO SDK timestamps reject conversion overflow without losing the raw value", "[ego][timing]")
{
    constexpr auto maximum_us = static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / 1000);
    CHECK(plugins::ego::sdk_timestamp_ns(0) == 0);
    CHECK(plugins::ego::sdk_timestamp_ns(123456789) == 123456789000);
    CHECK(plugins::ego::sdk_timestamp_ns(maximum_us) == static_cast<int64_t>(maximum_us) * 1000);
    for (const auto invalid_us : { maximum_us + 1, std::numeric_limits<uint64_t>::max() })
    {
        bool rejected = false;
        try
        {
            plugins::ego::sdk_timestamp_ns(invalid_us);
        }
        catch (const std::overflow_error& error)
        {
            rejected = true;
            CHECK(std::string(error.what()) ==
                  "EGO SDK timestamp_us=" + std::to_string(invalid_us) + " exceeds int64 nanosecond range");
        }
        CHECK(rejected);
    }
}

TEST_CASE("EGO missing timing model preserves callback times and unverified provenance", "[ego][timing]")
{
    plugins::ego::TimeAlignment alignment;
    alignment.configure(identity());
    alignment.begin_epoch(0);
    CHECK_FALSE(alignment.observe_left(100, 1000));
    CHECK(alignment.sample_time("gyro", 500, 9000) == 9000);
    const auto json = alignment.snapshot();
    CHECK(json.find("\"status\":\"unverified\"") != std::string::npos);
    CHECK(json.find("host_arrival_fallback") != std::string::npos);
    CHECK(json.find("\"serial\":\"S1\"") != std::string::npos);
    CHECK(json.find("\"sdk_binary_sha256\":\"" + std::string(64, 'a') + "\"") != std::string::npos);
    CHECK_FALSE(alignment.has_update());
}

TEST_CASE("EGO relative profile uses one estimated host anchor and resets it on recovery", "[ego][timing]")
{
    ProfileFile file;
    plugins::ego::TimeAlignment alignment(file.path.string());
    alignment.configure(identity());
    alignment.begin_epoch(0);
    CHECK(alignment.sample_time("gyro", 1000, 10000) == 10000);
    REQUIRE(alignment.observe_left(100000, 1000000));
    CHECK_FALSE(alignment.observe_left(200000, 9000000));
    CHECK(alignment.sample_time("left_rgb", 200000, 8000000) == 1100000);
    CHECK(alignment.sample_time("right_rgb", 201000, 9000000) == 1100000);
    CHECK(alignment.sample_time("accel", 202000, 1200000) == 1100000);
    CHECK(alignment.sample_time("gyro", 203000, 1300000) == 1100000);
    CHECK(alignment.sample_time("audio", 203000, 1400000) == 1400000);
    const auto json = alignment.snapshot();
    CHECK(json.find("left_arrival_estimate") != std::string::npos);
    CHECK(json.find("\"host_absolute_accuracy\":\"unverified\"") != std::string::npos);
    CHECK(json.find("first_mapped_raw_ns") != std::string::npos);
    alignment.begin_epoch(1);
    CHECK(alignment.sample_time("gyro", 500, 2000000) == 2000000);
    REQUIRE(alignment.observe_left(1000, 3000000));
    CHECK(alignment.sample_time("gyro", 104000, 3200000) == 3100000);
    CHECK_THROWS(alignment.sample_time("gyro", 103000, 3300000));
}

TEST_CASE("EGO capture timing input rejects untrusted or recording-bound models", "[ego][timing]")
{
    std::string invalid;
    SECTION("offline anchors")
    {
        invalid = replace(kProfile, "capture_profile", "recording");
    }
    SECTION("unverified")
    {
        invalid = replace(kProfile, "\"status\":\"validated\"", "\"status\":\"unverified\"");
    }
    SECTION("boolean integer")
    {
        invalid = replace(kProfile, "\"offset_ns\":0", "\"offset_ns\":false");
    }
    SECTION("missing independent evidence")
    {
        invalid = replace(kProfile, "[\"synthetic\"]", "[]");
    }
    SECTION("zero validation duration")
    {
        invalid = replace(kProfile, "1800000000000", "0");
    }
    SECTION("validation shorter than fifteen minutes")
    {
        invalid = replace(kProfile, "1800000000000", "899999999999");
    }
    SECTION("negative scale")
    {
        invalid = replace(kProfile, "\"scale\":1", "\"scale\":-1");
    }
    SECTION("nonunit scale")
    {
        invalid = replace(kProfile, "\"right_rgb\":{\"offset_ns\":-1000,\"scale\":1}",
                          "\"right_rgb\":{\"offset_ns\":-1000,\"scale\":1.00001}");
    }
    SECTION("missing SDK binary identity")
    {
        invalid = replace(kProfile, ",\n\"sdk_binary_sha256\":\"" + std::string(64, 'a') + "\"", "");
    }
    SECTION("malformed SDK binary digest")
    {
        invalid = replace(kProfile, std::string(64, 'a'), "not-a-digest");
    }
    SECTION("zero allowed capture duration")
    {
        invalid = replace(kProfile, "\"max_capture_duration_ns\":900000000000", "\"max_capture_duration_ns\":0");
    }
    SECTION("allowed capture duration exceeds evidence")
    {
        invalid =
            replace(kProfile, "\"max_capture_duration_ns\":900000000000", "\"max_capture_duration_ns\":1800000000001");
    }
    SECTION("duplicate keys")
    {
        invalid = replace(kProfile, "\"offset_ns\":0", "\"offset_ns\":10,\"offset_ns\":0");
    }
    SECTION("validation error exceeds half a frame")
    {
        invalid = replace(kProfile, "\"max_abs_error_ns\":1000", "\"max_abs_error_ns\":20000000");
    }
    SECTION("fractional profile integer")
    {
        invalid = replace(kProfile, "\"width\":1600", "\"width\":1600.5");
    }
    SECTION("oversized")
    {
        invalid = std::string(8193, ' ');
    }
    ProfileFile file(invalid);
    CHECK_THROWS(plugins::ego::TimeAlignment(file.path.string()));
}

TEST_CASE("EGO reusable timing validation accepts exactly fifteen minutes", "[ego][timing]")
{
    ProfileFile accepted(
        replace(kProfile, "\"held_out_duration_ns\":1800000000000", "\"held_out_duration_ns\":900000000000"));
    CHECK_NOTHROW(plugins::ego::validate_time_alignment_profile(accepted.path.string()));
    ProfileFile rejected(
        replace(kProfile, "\"held_out_duration_ns\":1800000000000", "\"held_out_duration_ns\":899999999999"));
    bool was_rejected = false;
    try
    {
        plugins::ego::validate_time_alignment_profile(rejected.path.string());
    }
    catch (const std::invalid_argument& error)
    {
        was_rejected = true;
        CHECK(std::string(error.what()) ==
              "EGO capture profile requires at least 15 minutes of independent held-out validation");
    }
    CHECK(was_rejected);
}

TEST_CASE("EGO capture timing profile binding prevents device and mode substitution", "[ego][timing]")
{
    ProfileFile file;
    plugins::ego::TimeAlignment alignment(file.path.string());
    auto selected = identity();
    SECTION("serial")
    {
        selected.serial = "other";
    }
    SECTION("firmware")
    {
        selected.firmware = "1.1.2";
    }
    SECTION("SDK")
    {
        selected.sdk_version = "2.10.0";
    }
    SECTION("SDK binary with the same version")
    {
        selected.sdk_binary_sha256 = std::string(64, 'b');
    }
    SECTION("video profile")
    {
        selected.width = 1280;
    }
    SECTION("IMU profile")
    {
        selected.imu_rate_hz = 400;
    }
    SECTION("audio profile")
    {
        selected.audio_channels = 2;
    }
    CHECK_THROWS(alignment.configure(selected));
}

TEST_CASE("EGO loaded SDK provenance resolves the linked version symbol", "[ego][timing]")
{
    const auto digest = plugins::ego::loaded_sdk_binary_sha256();
    CHECK(digest.size() == 64);
    CHECK(digest.find_first_not_of("0123456789abcdef") == std::string::npos);
    CHECK(plugins::ego::loaded_sdk_binary_sha256() == digest);
}

TEST_CASE("EGO negative delay begins mapping despite a later arrival fallback", "[ego][timing]")
{
    ProfileFile file(replace(kProfile, "\"gyro\":{\"offset_ns\":-3000", "\"gyro\":{\"offset_ns\":-20000000"));
    plugins::ego::TimeAlignment alignment(file.path.string());
    alignment.configure(identity());
    alignment.begin_epoch(0);
    CHECK(alignment.sample_time("gyro", 499000000, 1040000000) == 1040000000);
    REQUIRE(alignment.observe_left(500000000, 1000000000));
    alignment.snapshot();
    CHECK(alignment.sample_time("gyro", 519000000, 1050000000) == 1050000000);
    CHECK_FALSE(alignment.has_update());
    CHECK(alignment.sample_time("gyro", 521000000, 1060000000) == 1001000000);
    CHECK(alignment.has_update());
    CHECK(alignment.snapshot().find("\"first_mapped_raw_ns\":{\"gyro\":521000000}") != std::string::npos);
    CHECK(alignment.sample_time("gyro", 522000000, 1070000000) == 1002000000);
    CHECK_THROWS(alignment.sample_time("gyro", 520500000, 1080000000));
}

TEST_CASE("EGO capture profile expiry preserves arrival fallback and a bounded offline interval", "[ego][timing]")
{
    ProfileFile file(
        replace(kProfile, "\"max_capture_duration_ns\":900000000000", "\"max_capture_duration_ns\":100000000"));
    plugins::ego::TimeAlignment alignment(file.path.string());
    alignment.configure(identity());
    alignment.begin_epoch(0);
    REQUIRE(alignment.observe_left(1000000000, 2000000000));
    CHECK(alignment.sample_time("left_rgb", 1050000000, 2100000000) == 2050000000);
    CHECK(alignment.sample_time("gyro", 1090003000, 2110000000) == 2090000000);
    alignment.snapshot();
    CHECK(alignment.sample_time("gyro", 1120003000, 2200000000) == 2200000000);
    CHECK_FALSE(alignment.has_update());
    CHECK(alignment.sample_time("left_rgb", 1100000000, 2250000000) == 2250000000);
    CHECK(alignment.has_update());
    const auto expired = alignment.snapshot();
    CHECK(expired.find("\"status\":\"expired\"") != std::string::npos);
    CHECK(expired.find("\"valid_from_ns\":1000000000") != std::string::npos);
    CHECK(expired.find("\"valid_to_ns\":1100000000") != std::string::npos);
    CHECK(expired.find("\"expired_at_reference_ns\":1100000000") != std::string::npos);
    CHECK(expired.find("expired_capture_profile_host_arrival_fallback") != std::string::npos);
    CHECK(alignment.sample_time("gyro", 1110003000, 2300000000) == 2300000000);
    alignment.begin_epoch(1);
    REQUIRE(alignment.observe_left(10000000, 3000000000));
    CHECK(alignment.sample_time("gyro", 20003000, 3100000000) == 3010000000);
    CHECK(alignment.snapshot().find("\"status\":\"active\"") != std::string::npos);
}

TEST_CASE("EGO stereo timing profile does not require disabled IMU streams", "[ego][timing]")
{
    const auto stereo =
        replace(replace(kProfile, "\"imu_rate_hz\":1000", "\"imu_rate_hz\":0"),
                ",\n\"accel\":{\"offset_ns\":-2000,\"scale\":1},\"gyro\":{\"offset_ns\":-3000,\"scale\":1}", "");
    ProfileFile file(stereo);
    plugins::ego::TimeAlignment alignment(file.path.string());
    auto selected = identity();
    selected.imu_rate_hz = 0;
    REQUIRE_NOTHROW(alignment.configure(selected));
    CHECK(alignment.observe_left(100, 1000));
    CHECK(alignment.sample_time("right_rgb", 2100, 9000) == 2000);
    selected.imu_rate_hz = 1000;
    CHECK_THROWS(alignment.configure(selected));
}

TEST_CASE("EGO audio retained indices expose observed gaps without claiming physical continuity", "[ego][timing]")
{
    plugins::ego::TimeAlignment alignment;
    alignment.configure(identity());
    alignment.begin_epoch(0);
    const auto first = alignment.audio_timing(1000000000, 480, 48000);
    CHECK(first.segment_id == 0);
    CHECK(first.first_sample_index == 0);
    CHECK_FALSE(first.continuity_known);
    CHECK_FALSE(first.discontinuity);
    const auto next = alignment.audio_timing(1010000000, 480, 48000);
    CHECK(next.segment_id == 0);
    CHECK(next.first_sample_index == 480);
    const auto gap = alignment.audio_timing(1300000000, 480, 48000);
    CHECK(gap.segment_id == 1);
    CHECK(gap.first_sample_index == 0);
    CHECK(gap.discontinuity);
    alignment.begin_epoch(1);
    const auto recovered = alignment.audio_timing(500, 480, 48000);
    CHECK(recovered.segment_id == 2);
    CHECK(recovered.first_sample_index == 0);
    CHECK(recovered.discontinuity);
    CHECK_FALSE(recovered.continuity_known);
    CHECK_THROWS(alignment.audio_timing(600, 0, 48000));
}
