// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <ego_camera/calibration.hpp>
#include <ego_camera/cancellation.hpp>
#include <ego_camera/ego_camera.hpp>
#include <ego_mcap/checked_write.hpp>
#include <schema/orbbec_ego_camera_bfbs_generated.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#    include <sys/resource.h>

#    include <csignal>
#endif

namespace
{

class RecordingMetadataSink final : public plugins::ego::IMetadataSink
{
public:
    void on_frame_metadata(const plugins::ego::CapturedFrame& frame) override
    {
        frames.push_back(frame);
    }
    std::vector<plugins::ego::CapturedFrame> frames;
};

class RecordingMediaSink final : public plugins::ego::IMetadataSink
{
public:
    void on_frame_metadata(const plugins::ego::CapturedFrame&) override
    {
    }
    void on_encoded_video_frame(const plugins::ego::CapturedFrame& frame) override
    {
        frames.push_back(frame);
    }
    std::vector<plugins::ego::CapturedFrame> frames;
};

void append_nal(std::vector<uint8_t>& bytes, std::initializer_list<uint8_t> header, const std::string& payload = {})
{
    bytes.insert(bytes.end(), { 0, 0, 0, 1 });
    bytes.insert(bytes.end(), header);
    bytes.insert(bytes.end(), payload.begin(), payload.end());
}

std::vector<uint8_t> parameterized_idr(core::OrbbecEgoPixelFormat format)
{
    std::vector<uint8_t> bytes;
    if (format == core::OrbbecEgoPixelFormat_H264)
    {
        append_nal(bytes, { 0x67, 0x01 });
        append_nal(bytes, { 0x68, 0x01 });
        append_nal(bytes, { 0x65, 0x01 });
    }
    else
    {
        append_nal(bytes, { 0x40, 0x01, 0x01 });
        append_nal(bytes, { 0x42, 0x01, 0x01 });
        append_nal(bytes, { 0x44, 0x01, 0x01 });
        append_nal(bytes, { 0x26, 0x01, 0x01 });
    }
    return bytes;
}

// Firmware payload strings retain vendor bytes; only API names use Ego.
void append_sei(std::vector<uint8_t>& bytes,
                core::OrbbecEgoPixelFormat format,
                const std::string& payload,
                bool suffix = false)
{
    if (format == core::OrbbecEgoPixelFormat_H264)
        append_nal(bytes, { 0x06, 0x05, static_cast<uint8_t>(payload.size()) }, payload);
    else
        append_nal(bytes,
                   { static_cast<uint8_t>(suffix ? 0x50 : 0x4e), 0x01, 0x05, static_cast<uint8_t>(payload.size()) },
                   payload);
    bytes.push_back(0x80);
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

} // namespace

TEST_CASE("EGO IMU YAML preserves the SDK transform direction and millimeters", "[ego][calibration]")
{
    const std::string yaml = R"(%YAML:1.0
---
cam0:
  T_cam_imu:
    - [0, -1, 0, 48] # IMU x points down in the camera
    - [1, 0, 0, 5]
    - [0, 0, 1, -21.84]
    - [0, 0, 0, 1]
  resolution: [1600, 1300]
cam1:
  T_cam_imu: [[1, 0, 0, 999], [0, 1, 0, 999], [0, 0, 1, 999], [0, 0, 0, 1]]
imu0:
  C_gyro_i: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
)";
    core::OrbbecEgoCalibrationT calibration;
    calibration.raw_imu_yaml = yaml;
    calibration.color_left = std::make_shared<core::OrbbecEgoCameraIntrinsicsT>();
    calibration.color_left->width = 1600;
    calibration.color_left->height = 1300;
    const auto color_left = calibration.color_left;
    std::shared_ptr<core::OrbbecEgoExtrinsicsT> preserved;
    SECTION("both IMU fields are missing")
    {
    }
    SECTION("image profile does not change the IMU rigid transform")
    {
        calibration.color_left->width = 640;
        calibration.color_left->height = 480;
    }
    SECTION("SDK accelerometer extrinsics are preserved")
    {
        preserved = std::make_shared<core::OrbbecEgoExtrinsicsT>();
        preserved->translation_mm = { 7, 8, 9 };
        calibration.accel_to_left = preserved;
    }
    SECTION("SDK gyroscope extrinsics are preserved")
    {
        preserved = std::make_shared<core::OrbbecEgoExtrinsicsT>();
        preserved->translation_mm = { 7, 8, 9 };
        calibration.gyro_to_left = preserved;
    }
    REQUIRE(plugins::ego::populate_imu_calibration_from_yaml(yaml, calibration));
    REQUIRE(calibration.accel_to_left);
    REQUIRE(calibration.gyro_to_left);
    for (const auto& extrinsics : { calibration.accel_to_left, calibration.gyro_to_left })
    {
        if (extrinsics == preserved)
        {
            CHECK(extrinsics->translation_mm == std::vector<float>{ 7, 8, 9 });
            continue;
        }
        CHECK(extrinsics->rotation == std::vector<float>{ 0, -1, 0, 1, 0, 0, 0, 0, 1 });
        CHECK(extrinsics->translation_mm == std::vector<float>{ 48, 5, -21.84f });
    }
    CHECK(calibration.color_left == color_left);
    CHECK(calibration.raw_imu_yaml == yaml);
}

TEST_CASE("EGO IMU YAML rejects invalid transforms without partial updates", "[ego][calibration]")
{
    std::string yaml = "cam0:\n  T_cam_imu: [[0,-1,0,48], [1,0,0,5], [0,0,1,-21.84], [0,0,0,1]]\n";
    SECTION("malformed YAML")
    {
        yaml = "cam0: [";
    }
    SECTION("missing cam0 is never replaced with another camera")
    {
        yaml.replace(0, 4, "cam1");
    }
    SECTION("missing transform")
    {
        yaml = "cam0: {resolution: [1600,1300]}";
    }
    SECTION("three rows")
    {
        yaml = "cam0: {T_cam_imu: [[1,0,0,1], [0,1,0,2], [0,0,1,3]]}";
    }
    SECTION("ragged matrix")
    {
        yaml = "cam0: {T_cam_imu: [[1,0,0], [0,1,0,2], [0,0,1,3], [0,0,0,1]]}";
    }
    SECTION("non-numeric element")
    {
        yaml = "cam0: {T_cam_imu: [[1,0,0,bad], [0,1,0,2], [0,0,1,3], [0,0,0,1]]}";
    }
    SECTION("infinite translation")
    {
        yaml = "cam0: {T_cam_imu: [[1,0,0,.inf], [0,1,0,2], [0,0,1,3], [0,0,0,1]]}";
    }
    SECTION("NaN rotation")
    {
        yaml = "cam0: {T_cam_imu: [[.nan,0,0,1], [0,1,0,2], [0,0,1,3], [0,0,0,1]]}";
    }
    SECTION("float overflow")
    {
        yaml = "cam0: {T_cam_imu: [[1,0,0,1e100], [0,1,0,2], [0,0,1,3], [0,0,0,1]]}";
    }
    SECTION("non-homogeneous bottom row")
    {
        yaml = "cam0: {T_cam_imu: [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1]]}";
    }
    SECTION("scaled rotation")
    {
        yaml = "cam0: {T_cam_imu: [[2,0,0,1], [0,1,0,2], [0,0,1,3], [0,0,0,1]]}";
    }
    SECTION("reflection")
    {
        yaml = "cam0: {T_cam_imu: [[-1,0,0,1], [0,1,0,2], [0,0,1,3], [0,0,0,1]]}";
    }
    SECTION("duplicate camera")
    {
        yaml += yaml;
    }
    SECTION("duplicate transform")
    {
        yaml += "  T_cam_imu: [[1,0,0,1], [0,1,0,2], [0,0,1,3], [0,0,0,1]]\n";
    }
    SECTION("multiple documents")
    {
        yaml += "---\n" + yaml;
    }
    core::OrbbecEgoCalibrationT calibration;
    calibration.raw_imu_yaml = yaml;
    calibration.color_left = std::make_shared<core::OrbbecEgoCameraIntrinsicsT>();
    calibration.color_left->width = 1600;
    calibration.color_left->height = 1300;
    calibration.accel_to_left = std::make_shared<core::OrbbecEgoExtrinsicsT>();
    calibration.accel_to_left->translation_mm = { 7, 8, 9 };
    const auto accel = calibration.accel_to_left;
    CHECK_FALSE(plugins::ego::populate_imu_calibration_from_yaml(yaml, calibration));
    CHECK(calibration.accel_to_left == accel);
    CHECK(calibration.accel_to_left->translation_mm == std::vector<float>{ 7, 8, 9 });
    CHECK_FALSE(calibration.gyro_to_left);
    CHECK(calibration.raw_imu_yaml == yaml);
}

TEST_CASE("EGO FrameSink writes raw MJPEG and forwards metadata", "[ego][writer][metadata]")
{
    const auto output = std::filesystem::temp_directory_path() / "isaacteleop_ego_camera_test.mjpg";
    auto metadata_sink = std::make_unique<RecordingMetadataSink>();
    auto* metadata_sink_ptr = metadata_sink.get();
    {
        plugins::ego::FrameSink sink(
            { { core::OrbbecEgoCameraStream_ColorLeft, output.string() } }, std::move(metadata_sink));
        plugins::ego::CapturedFrame frame;
        frame.metadata.stream = core::OrbbecEgoCameraStream_ColorLeft;
        frame.metadata.sequence_number = 42;
        frame.metadata.width = 1280;
        frame.metadata.height = 720;
        frame.metadata.fps = 30;
        frame.metadata.pixel_format = core::OrbbecEgoPixelFormat_Mjpg;
        frame.encoded_data = { 0xff, 0xd8, 0x01, 0x02, 0xff, 0xd9 };
        frame.sample_time_local_common_clock_ns = 100;
        frame.sample_time_raw_device_clock_ns = 200;
        sink.on_frame(frame);
        REQUIRE(metadata_sink_ptr->frames.size() == 1);
        REQUIRE(metadata_sink_ptr->frames.front().metadata.sequence_number == 42);
    }

    std::ifstream input(output, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    REQUIRE(bytes.size() == 6);
    REQUIRE(static_cast<uint8_t>(bytes.front()) == 0xff);
    REQUIRE(static_cast<uint8_t>(bytes.back()) == 0xd9);
    std::filesystem::remove(output);
}

TEST_CASE("EGO capture refuses existing sidecars before creating sibling outputs", "[ego][writer]")
{
    const auto existing = std::filesystem::temp_directory_path() / "ego_existing_sidecar_test";
    const auto sibling = std::filesystem::temp_directory_path() / "ego_new_sidecar_test";
    {
        std::ofstream file(existing);
        file << "preserve";
    }
    plugins::ego::CaptureConfig config;
    std::vector<plugins::ego::StreamConfig> streams{ { core::OrbbecEgoCameraStream_ColorLeft, sibling.string() } };
    SECTION("video")
    {
        streams.push_back({ core::OrbbecEgoCameraStream_ColorRight, existing.string() });
    }
    SECTION("audio")
    {
        config.audio_output = existing.string();
    }
    SECTION("calibration")
    {
        config.calibration_output = existing.string();
    }
    REQUIRE_THROWS_AS(plugins::ego::create_frame_sink(streams, config), std::invalid_argument);
    CHECK(read_bytes(existing) == std::vector<uint8_t>{ 'p', 'r', 'e', 's', 'e', 'r', 'v', 'e' });
    CHECK_FALSE(std::filesystem::exists(sibling));
    std::filesystem::remove(existing);
}

TEST_CASE("EGO capture rejects aliased outputs before creating files", "[ego][writer]")
{
    const auto output = std::filesystem::temp_directory_path() / "ego_aliased_output_test";
    plugins::ego::CaptureConfig config;
    std::vector<plugins::ego::StreamConfig> streams{ { core::OrbbecEgoCameraStream_ColorLeft, output.string() } };
    SECTION("two cameras")
    {
        streams.push_back({ core::OrbbecEgoCameraStream_ColorRight, output.string() });
    }
    SECTION("audio")
    {
        config.audio_output = (output.parent_path() / "." / output.filename()).string();
    }
    SECTION("calibration")
    {
        config.calibration_output = output.string();
    }
    SECTION("archive")
    {
        config.mcap_filename = output.string();
    }
    REQUIRE_THROWS_AS(plugins::ego::create_frame_sink(streams, config), std::invalid_argument);
    CHECK_FALSE(std::filesystem::exists(output));
}

TEST_CASE("EGO FrameSink preserves H264 and H265 elementary stream bytes", "[ego][writer]")
{
    for (const auto format : { core::OrbbecEgoPixelFormat_H264, core::OrbbecEgoPixelFormat_H265 })
    {
        const auto suffix = format == core::OrbbecEgoPixelFormat_H264 ? ".h264" : ".h265";
        const auto output = std::filesystem::temp_directory_path() / (std::string("isaacteleop_ego") + suffix);
        size_t expected_size = 0;
        {
            plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, output.string() };
            stream.pixel_format = format;
            plugins::ego::FrameSink sink({ stream });
            plugins::ego::CapturedFrame frame;
            frame.metadata.stream = stream.camera;
            frame.metadata.pixel_format = format;
            if (format == core::OrbbecEgoPixelFormat_H264)
                frame.encoded_data = { 0, 0, 0, 1, 0x67, 0x01, 0, 0, 0, 1, 0x68, 0x01, 0, 0, 0, 1, 0x65, 0x01 };
            else
                frame.encoded_data = { 0, 0, 0, 1, 0x40, 0x01, 0, 0, 0, 1, 0x42, 0x01,
                                       0, 0, 0, 1, 0x44, 0x01, 0, 0, 0, 1, 0x26, 0x01 };
            expected_size = frame.encoded_data.size();
            sink.on_frame(frame);
        }
        std::ifstream input(output, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        REQUIRE(bytes.size() == expected_size);
        CHECK(bytes[3] == 1);
        std::filesystem::remove(output);
    }
}

TEST_CASE("EGO embedded FrameSink does not require sidecar paths", "[ego][writer][mcap]")
{
    auto media_sink = std::make_unique<RecordingMediaSink>();
    auto* media_sink_ptr = media_sink.get();
    plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, "" };
    stream.pixel_format = core::OrbbecEgoPixelFormat_H264;
    plugins::ego::FrameSink sink({ stream }, std::move(media_sink), false);
    plugins::ego::CapturedFrame frame;
    frame.metadata.stream = stream.camera;
    frame.metadata.pixel_format = stream.pixel_format;
    frame.metadata.sequence_number = 1;
    frame.encoded_data = { 0, 0, 0, 1, 0x67, 1, 0, 0, 0, 1, 0x68, 1, 0, 0, 0, 1, 0x65, 1 };
    sink.on_frame(frame);
    REQUIRE(media_sink_ptr->frames.size() == 1);
    CHECK(media_sink_ptr->frames.front().metadata.encoded_bytes == frame.encoded_data.size());
}

TEST_CASE("EGO stream validation delegates encoded FPS support to SDK profile selection", "[ego][profile]")
{
    plugins::ego::CaptureConfig capture;
    plugins::ego::StreamConfig h264{ core::OrbbecEgoCameraStream_ColorLeft, "unused.h264" };
    h264.pixel_format = core::OrbbecEgoPixelFormat_H264;
    h264.fps = 60;
    REQUIRE_NOTHROW(plugins::ego::validate_stream_config(h264, capture));

    plugins::ego::StreamConfig h265{ core::OrbbecEgoCameraStream_ColorRight, "unused.h265" };
    h265.pixel_format = core::OrbbecEgoPixelFormat_H265;
    capture.fps = 60;
    REQUIRE_NOTHROW(plugins::ego::validate_stream_config(h265, capture));

    capture.fps = 0;
    REQUIRE_NOTHROW(plugins::ego::validate_stream_config(h265, capture));
}

TEST_CASE("EGO resolved-profile certification closes automatic FPS bypass", "[ego][profile]")
{
    plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, "unused.h264" };
    stream.pixel_format = core::OrbbecEgoPixelFormat_H264;
    stream.fps = 0;
    CHECK_NOTHROW(plugins::ego::validate_resolved_stream_config(stream, 30));
    // This target's Catch2 macro subset omits CHECK_THROWS_WITH, so inspect the exception explicitly.
    bool rejected = false;
    try
    {
        plugins::ego::validate_resolved_stream_config(stream, 60);
    }
    catch (const std::invalid_argument& error)
    {
        rejected = true;
        CHECK(std::string(error.what()) ==
              "resolved encoded profile is 60 FPS, but this build certifies encoded recording only through 30 FPS; "
              "no profile fallback was attempted");
    }
    CHECK(rejected);

    stream.pixel_format = core::OrbbecEgoPixelFormat_H265;
    CHECK_THROWS(plugins::ego::validate_resolved_stream_config(stream, 60));
    stream.pixel_format = core::OrbbecEgoPixelFormat_Mjpg;
    CHECK_NOTHROW(plugins::ego::validate_resolved_stream_config(stream, 60));
}

TEST_CASE("EGO FrameSink resumes compressed recording at an IDR after a sequence gap", "[ego][writer]")
{
    const auto output = std::filesystem::temp_directory_path() / "isaacteleop_ego_gap.h264";
    plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, output.string() };
    stream.pixel_format = core::OrbbecEgoPixelFormat_H264;
    const std::vector<uint8_t> idr = { 0, 0, 0, 1, 0x67, 0x01, 0, 0, 0, 1, 0x68, 0x01, 0, 0, 0, 1, 0x65, 0x01 };
    const std::vector<uint8_t> p_frame = { 0, 0, 0, 1, 0x41, 0x01 };
    {
        plugins::ego::FrameSink sink({ stream });
        plugins::ego::CapturedFrame frame;
        frame.metadata.stream = stream.camera;
        frame.metadata.pixel_format = stream.pixel_format;
        frame.metadata.sequence_number = 1;
        frame.encoded_data = idr;
        sink.on_frame(frame);

        frame.metadata.sequence_number = 3;
        frame.encoded_data = p_frame;
        sink.on_frame(frame);

        frame.metadata.sequence_number = 4;
        sink.on_frame(frame);

        frame.metadata.sequence_number = 5;
        frame.encoded_data = idr;
        sink.on_frame(frame);
    }
    std::ifstream input(output, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    REQUIRE(bytes.size() == idr.size() * 2);
    std::filesystem::remove(output);
}

TEST_CASE("EGO FrameSink starts every capture epoch at a parameterized IDR", "[ego][writer][reconnect]")
{
    const auto output = std::filesystem::temp_directory_path() / "isaacteleop_ego_epoch.h264";
    plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, output.string() };
    stream.pixel_format = core::OrbbecEgoPixelFormat_H264;
    const auto idr = parameterized_idr(stream.pixel_format);
    const std::vector<uint8_t> p_frame = { 0, 0, 0, 1, 0x41, 0x01 };
    {
        plugins::ego::FrameSink sink({ stream });
        plugins::ego::CapturedFrame frame;
        frame.metadata.stream = stream.camera;
        frame.metadata.pixel_format = stream.pixel_format;
        frame.metadata.sequence_number = 100;
        frame.encoded_data = idr;
        sink.on_frame(frame);
        frame.metadata.sequence_number = 101;
        frame.encoded_data = p_frame;
        sink.on_frame(frame);

        sink.begin_capture_epoch();
        frame.metadata.sequence_number = 0;
        sink.on_frame(frame);
        frame.metadata.sequence_number = 1;
        frame.encoded_data = idr;
        sink.on_frame(frame);
    }
    auto expected = idr;
    expected.insert(expected.end(), p_frame.begin(), p_frame.end());
    expected.insert(expected.end(), idr.begin(), idr.end());
    REQUIRE(read_bytes(output) == expected);
    std::filesystem::remove(output);
}

TEST_CASE("EGO FrameSink removes legacy and firmware 1.1.1 timestamp SEI", "[ego][writer]")
{
    for (const auto format : { core::OrbbecEgoPixelFormat_H264, core::OrbbecEgoPixelFormat_H265 })
    {
        const auto frame_field = format == core::OrbbecEgoPixelFormat_H264 ? "frameId" : "frame_seq";
        const std::vector<std::string> timestamp_payloads = {
            "ORBBEC,EGO_",
            "EGO0000000000_L,timestamp_us=1788430979164168," + std::string(frame_field) + "=1",
            "EGO0000000000_R,timestamp_us=1788430979180835," + std::string(frame_field) + "=2",
        };
        for (size_t index = 0; index < timestamp_payloads.size(); ++index)
        {
            const auto codec = format == core::OrbbecEgoPixelFormat_H264 ? "h264" : "h265";
            const auto output = std::filesystem::temp_directory_path() /
                                (std::string("isaacteleop_ego_timestamp_") + codec + "_" + std::to_string(index));
            const auto expected = parameterized_idr(format);
            auto encoded = expected;
            append_sei(encoded, format, timestamp_payloads[index]);
            if (format == core::OrbbecEgoPixelFormat_H265)
                append_sei(encoded, format, timestamp_payloads[index], true);

            {
                auto media_sink = std::make_unique<RecordingMediaSink>();
                auto* media_sink_ptr = media_sink.get();
                plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, output.string() };
                stream.pixel_format = format;
                plugins::ego::FrameSink sink({ stream }, std::move(media_sink));
                plugins::ego::CapturedFrame frame;
                frame.metadata.stream = stream.camera;
                frame.metadata.pixel_format = stream.pixel_format;
                frame.metadata.sequence_number = 1;
                frame.encoded_data = std::move(encoded);
                sink.on_frame(frame);
                REQUIRE(media_sink_ptr->frames.size() == 1);
                CHECK(media_sink_ptr->frames.front().encoded_data == expected);
                CHECK(media_sink_ptr->frames.front().metadata.encoded_bytes == expected.size());
            }

            REQUIRE(read_bytes(output) == expected);
            std::filesystem::remove(output);
        }
    }
}

TEST_CASE("EGO FrameSink preserves non-timestamp H264 and H265 SEI", "[ego][writer]")
{
    for (const auto format : { core::OrbbecEgoPixelFormat_H264, core::OrbbecEgoPixelFormat_H265 })
    {
        const auto frame_field = format == core::OrbbecEgoPixelFormat_H264 ? "frameId" : "frame_seq";
        const std::vector<std::string> ordinary_payloads = {
            "ORBBEC,CAMERA_INFO",
            "EGO0000000000_L,camera_info=stereo",
            "timestamp_us=1788430979164168," + std::string(frame_field) + "=1",
            "EGO0000000000_L," + std::string(frame_field) + "=1,timestamp_us=1788430979164168",
            "EGO0000000000_L,timestamp_us=1788430979164168," +
                std::string(format == core::OrbbecEgoPixelFormat_H264 ? "frame_seq" : "frameId") + "=1",
        };
        const auto codec = format == core::OrbbecEgoPixelFormat_H264 ? "h264" : "h265";
        const auto output =
            std::filesystem::temp_directory_path() / (std::string("isaacteleop_ego_ordinary_sei_") + codec);
        auto encoded = parameterized_idr(format);
        for (const auto& payload : ordinary_payloads)
            append_sei(encoded, format, payload);
        if (format == core::OrbbecEgoPixelFormat_H265)
            append_sei(encoded, format, ordinary_payloads.front(), true);

        {
            auto media_sink = std::make_unique<RecordingMediaSink>();
            auto* media_sink_ptr = media_sink.get();
            plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, output.string() };
            stream.pixel_format = format;
            plugins::ego::FrameSink sink({ stream }, std::move(media_sink));
            plugins::ego::CapturedFrame frame;
            frame.metadata.stream = stream.camera;
            frame.metadata.pixel_format = stream.pixel_format;
            frame.metadata.sequence_number = 1;
            frame.encoded_data = encoded;
            sink.on_frame(frame);
            REQUIRE(media_sink_ptr->frames.size() == 1);
            CHECK(media_sink_ptr->frames.front().encoded_data == encoded);
        }

        REQUIRE(read_bytes(output) == encoded);
        std::filesystem::remove(output);
    }
}

TEST_CASE("EGO FrameSink observers share the canonical recording gate", "[ego][writer][preview]")
{
    for (const auto format : { core::OrbbecEgoPixelFormat_H264, core::OrbbecEgoPixelFormat_H265 })
    {
        plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, "" };
        stream.pixel_format = format;
        auto media = std::make_unique<RecordingMediaSink>();
        auto* media_ptr = media.get();
        plugins::ego::FrameSink sink({ stream }, std::move(media), false);
        std::vector<plugins::ego::CapturedFrame> observed;
        const auto observe = [&](const plugins::ego::CapturedFrame& accepted)
        {
            REQUIRE(!media_ptr->frames.empty());
            CHECK(accepted.encoded_data == media_ptr->frames.back().encoded_data);
            CHECK(accepted.metadata.encoded_bytes == accepted.encoded_data.size());
            CHECK(accepted.sample_time_local_common_clock_ns == 100);
            CHECK(accepted.sample_time_raw_device_clock_ns == 200);
            observed.push_back(accepted);
        };
        std::vector<uint8_t> predicted;
        append_nal(predicted, format == core::OrbbecEgoPixelFormat_H264 ?
                                  std::initializer_list<uint8_t>{ 0x41, 0x01 } :
                                  std::initializer_list<uint8_t>{ 0x02, 0x01, 0x01 });
        auto canonical_idr = parameterized_idr(format);
        append_sei(canonical_idr, format, "ORBBEC,CAMERA_INFO");
        auto raw_idr = canonical_idr;
        append_sei(raw_idr, format, "ORBBEC,EGO_");
        std::vector<uint8_t> timestamp_only;
        append_sei(timestamp_only, format, "ORBBEC,EGO_");

        plugins::ego::CapturedFrame frame;
        frame.metadata.stream = stream.camera;
        frame.metadata.pixel_format = format;
        frame.metadata.capture_epoch = 7;
        frame.sample_time_local_common_clock_ns = 100;
        frame.sample_time_raw_device_clock_ns = 200;
        frame.encoded_data = predicted;
        sink.on_frame(frame, observe);
        CHECK(observed.empty());

        frame.metadata.sequence_number = 1;
        frame.encoded_data = raw_idr;
        sink.on_frame(frame, observe);
        REQUIRE(observed.size() == 1);
        CHECK(observed.back().encoded_data == canonical_idr);
        CHECK(frame.encoded_data == raw_idr);

        frame.metadata.sequence_number = 2;
        frame.encoded_data = timestamp_only;
        sink.on_frame(frame, observe);
        CHECK(observed.size() == 1);
        frame.metadata.sequence_number = 3;
        frame.encoded_data = predicted;
        sink.on_frame(frame, observe);
        REQUIRE(observed.size() == 2);
        CHECK(observed.back().encoded_data == predicted);

        frame.metadata.sequence_number = 5;
        sink.on_frame(frame, observe);
        frame.metadata.sequence_number = 6;
        sink.on_frame(frame, observe);
        CHECK(observed.size() == 2);
        frame.metadata.sequence_number = 7;
        frame.encoded_data = raw_idr;
        sink.on_frame(frame, observe);
        REQUIRE(observed.size() == 3);
        CHECK(observed.back().metadata.sequence_number == 7);

        sink.begin_capture_epoch();
        frame.metadata.capture_epoch = 8;
        frame.metadata.sequence_number = 0;
        frame.encoded_data = predicted;
        sink.on_frame(frame, observe);
        CHECK(observed.size() == 3);
        frame.metadata.sequence_number = 1;
        frame.encoded_data = raw_idr;
        sink.on_frame(frame, observe);
        REQUIRE(observed.size() == 4);
        CHECK(observed.back().metadata.capture_epoch == 8);
        CHECK(observed.back().encoded_data == canonical_idr);
        CHECK(media_ptr->frames.size() == observed.size());
    }
}

TEST_CASE("EGO FrameSink observers work without a metadata sink", "[ego][writer][preview]")
{
    plugins::ego::FrameSink sink({ { core::OrbbecEgoCameraStream_ColorLeft, "" } }, nullptr, false);
    plugins::ego::CapturedFrame frame;
    frame.metadata.stream = core::OrbbecEgoCameraStream_ColorLeft;
    frame.metadata.sequence_number = 42;
    frame.encoded_data = { 0xff, 0xd8, 0xff, 0xd9 };
    size_t observations = 0;
    sink.on_frame(frame,
                  [&](const plugins::ego::CapturedFrame& accepted)
                  {
                      ++observations;
                      CHECK(accepted.metadata.sequence_number == 42);
                      CHECK(accepted.metadata.encoded_bytes == 4);
                      CHECK(accepted.encoded_data == frame.encoded_data);
                  });
    CHECK(observations == 1);
}

TEST_CASE("EGO local MCAP and SchemaPusher modes are mutually exclusive", "[ego][cli][mcap]")
{
    plugins::ego::CaptureConfig config;
    config.collection_prefix = "ego";
    config.mcap_filename = "metadata.mcap";
    REQUIRE_THROWS_AS(plugins::ego::create_frame_sink({}, config), std::invalid_argument);
}

TEST_CASE("EGO local MCAP is promoted only by an explicit successful close", "[ego][mcap][shutdown]")
{
    const auto base = std::filesystem::temp_directory_path() / "isaacteleop_ego_shutdown_test.mcap";
    const auto partial = std::filesystem::path(base.string() + ".partial");
    plugins::ego::CaptureConfig config;
    config.mcap_media_mode = plugins::ego::McapMediaMode::Embedded;
    plugins::ego::StreamConfig stream{ core::OrbbecEgoCameraStream_ColorLeft, "" };
    stream.pixel_format = core::OrbbecEgoPixelFormat_H264;

    std::filesystem::remove(base);
    std::filesystem::remove(partial);
    config.mcap_filename = base.string();
    {
        auto sink = plugins::ego::create_frame_sink({ stream }, config);
        sink->abort_metadata();
    }
    CHECK_FALSE(std::filesystem::exists(base));
    CHECK(std::filesystem::exists(partial));
    std::filesystem::remove(partial);

    {
        auto sink = plugins::ego::create_frame_sink({ stream }, config);
        CHECK_THROWS(sink->commit_metadata());
        sink->close_media();
        sink->close_metadata();
        CHECK_FALSE(std::filesystem::exists(base));
        CHECK(std::filesystem::exists(partial));
        sink->commit_metadata();
    }
    CHECK(std::filesystem::exists(base));
    CHECK_FALSE(std::filesystem::exists(partial));
    std::filesystem::remove(base);
}

TEST_CASE("EGO composite metadata rejects promotion after any close failure", "[ego][mcap][shutdown]")
{
    struct Sink final : plugins::ego::IMetadataSink
    {
        bool fail = false;
        bool closed = false;
        bool committed = false;
        void on_frame_metadata(const plugins::ego::CapturedFrame&) override
        {
        }
        void close() override
        {
            closed = true;
            if (fail)
                throw std::runtime_error("publisher close failed");
        }
        void commit() override
        {
            committed = true;
        }
    };
    auto publisher = std::make_unique<Sink>();
    auto archive = std::make_unique<Sink>();
    auto* publisher_ptr = publisher.get();
    auto* archive_ptr = archive.get();
    publisher->fail = true;
    std::vector<std::unique_ptr<plugins::ego::IMetadataSink>> children;
    children.push_back(std::move(publisher));
    children.push_back(std::move(archive));
    auto sink = plugins::ego::compose_metadata_sinks(std::move(children));
    CHECK_THROWS(sink->close());
    CHECK(publisher_ptr->closed);
    CHECK(archive_ptr->closed);
    CHECK_THROWS(sink->commit());
    CHECK_FALSE(publisher_ptr->committed);
    CHECK_FALSE(archive_ptr->committed);
}

TEST_CASE("EGO checked MCAP writes reject non-I/O writer errors", "[ego][mcap]")
{
    const auto path = std::filesystem::temp_directory_path() / "ego_closed_writer_test.mcap";
    mcap::McapWriter writer;
    REQUIRE(writer.open(path.string(), mcap::McapWriterOptions("ego")).ok());
    core::McapTrackerChannels<core::OrbbecEgoFrameMetadataRecord> channels(
        writer, "orbbec_ego_metadata", { "ColorLeft" });
    writer.close();
    const auto record =
        core::pack_record<core::OrbbecEgoFrameMetadataRecord>(nullptr, core::DeviceDataTimestamp(1, 2, 3));
    CHECK_THROWS_AS(ego_mcap::write_checked(writer, channels, 0, record), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("EGO startup waits respond to cancellation without waiting for readiness timeout", "[ego][shutdown]")
{
    std::atomic<bool> cancelled{ false };
    std::mutex mutex;
    std::condition_variable wake;
    std::thread stopper(
        [&]
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            cancelled = true;
        });
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex);
    CHECK_THROWS_AS(plugins::ego::wait_capture_ready(
                        wake, lock, std::chrono::seconds(10), [] { return false; }, [&] { return cancelled.load(); }),
                    plugins::ego::CaptureCancelled);
    stopper.join();
    CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
}

TEST_CASE("EGO abort after sealing cannot promote a readable partial MCAP", "[ego][mcap][shutdown]")
{
    const auto path = std::filesystem::temp_directory_path() / "ego_sealed_abort_test.mcap";
    std::filesystem::remove(path);
    plugins::ego::CaptureConfig config;
    config.mcap_filename = path.string();
    config.mcap_media_mode = plugins::ego::McapMediaMode::Embedded;
    auto sink = plugins::ego::create_frame_sink({ { core::OrbbecEgoCameraStream_ColorLeft, "" } }, config);
    sink->close_media();
    sink->close_metadata();
    sink->abort_metadata();
    CHECK_THROWS(sink->commit_metadata());
    CHECK_FALSE(std::filesystem::exists(path));
    CHECK(std::filesystem::exists(path.string() + ".partial"));
    std::filesystem::remove(path.string() + ".partial");
}

TEST_CASE("EGO capture preserves existing final and partial archives", "[ego][mcap][shutdown]")
{
    const auto path = std::filesystem::temp_directory_path() / "ego_existing_output_test.mcap";
    const auto partial = std::filesystem::path(path.string() + ".partial");
    std::filesystem::remove(path);
    std::filesystem::remove(partial);
    plugins::ego::CaptureConfig config;
    config.mcap_filename = path.string();
    config.mcap_media_mode = plugins::ego::McapMediaMode::Embedded;
    const std::vector<uint8_t> expected{ 'k', 'e', 'e', 'p' };
    SECTION("existing final")
    {
        std::ofstream(path) << "keep";
        CHECK_THROWS(plugins::ego::create_frame_sink({}, config));
        CHECK(read_bytes(path) == expected);
        CHECK_FALSE(std::filesystem::exists(partial));
    }
    SECTION("existing partial")
    {
        std::ofstream(partial) << "keep";
        CHECK_THROWS(plugins::ego::create_frame_sink({}, config));
        CHECK(read_bytes(partial) == expected);
        CHECK_FALSE(std::filesystem::exists(path));
    }
#if defined(__linux__)
    SECTION("dangling partial symlink")
    {
        std::filesystem::create_symlink(path, partial);
        CHECK_THROWS(plugins::ego::create_frame_sink({}, config));
        CHECK(std::filesystem::is_symlink(partial));
        CHECK_FALSE(std::filesystem::exists(path));
    }
#endif
    SECTION("final appears before commit")
    {
        auto sink = plugins::ego::create_frame_sink({}, config);
        sink->close_media();
        sink->close_metadata();
        std::ofstream(path) << "keep";
        CHECK_THROWS(sink->commit_metadata());
        CHECK(read_bytes(path) == expected);
        CHECK(std::filesystem::exists(partial));
    }
    std::filesystem::remove(path);
    std::filesystem::remove(partial);
}

TEST_CASE("EGO embedded MCAP preserves structured and media records through the final drain", "[ego][mcap]")
{
    const auto path = std::filesystem::temp_directory_path() / "ego_complete_fragment_test.mcap";
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".partial");
    plugins::ego::CaptureConfig config;
    config.mcap_filename = path.string();
    config.mcap_media_mode = plugins::ego::McapMediaMode::Embedded;
    auto sink = plugins::ego::create_frame_sink(
        { { core::OrbbecEgoCameraStream_ColorLeft, "" }, { core::OrbbecEgoCameraStream_ColorRight, "" } }, config);
    for (const auto camera : { core::OrbbecEgoCameraStream_ColorLeft, core::OrbbecEgoCameraStream_ColorRight })
    {
        for (uint64_t sequence = 0; sequence < 2; ++sequence)
        {
            plugins::ego::CapturedFrame frame;
            frame.metadata.stream = camera;
            frame.metadata.sequence_number = sequence;
            frame.metadata.capture_epoch = 3;
            frame.metadata.width = 1600;
            frame.metadata.height = 1300;
            frame.metadata.fps = 30;
            frame.encoded_data = { 0xff, 0xd8, 0xff, 0xd9 };
            frame.sample_time_local_common_clock_ns = 123 + sequence;
            frame.sample_time_raw_device_clock_ns = 456 + sequence;
            sink->on_frame(frame);
        }
    }
    auto* metadata = sink->metadata_sink();
    core::OrbbecEgoImuBatchT imu;
    imu.capture_epoch = 3;
    imu.sensor = core::OrbbecEgoImuSensor_Accel;
    metadata->on_imu_batch(imu, 123, 456);
    imu.sensor = core::OrbbecEgoImuSensor_Gyro;
    metadata->on_imu_batch(imu, 123, 456);
    core::OrbbecEgoAudioChunkT audio;
    audio.capture_epoch = 3;
    metadata->on_audio_chunk(audio, 123, 456);
    core::OrbbecEgoPcmAudioChunkT pcm;
    pcm.capture_epoch = 3;
    pcm.sample_rate_hz = 48'000;
    pcm.channel_count = 1;
    pcm.bits_per_sample = 16;
    pcm.sample_count = 1;
    pcm.pcm_data = { 1, 2 };
    metadata->on_pcm_audio_chunk(pcm, 123, 456);
    core::OrbbecEgoCalibrationT calibration;
    calibration.capture_epoch = 3;
    metadata->on_calibration(calibration, 123, 456);
    core::OrbbecEgoDeviceStateT state;
    state.capture_epoch = 3;
    state.device_uid = "fixture";
    state.observed_at_ns = 123;
    state.diagnostic_version = 2;
    core::OrbbecEgoCaptureSummaryT summary;
    summary.is_final = true;
    summary.queue_capacity = 4096;
    summary.queue_peak = 7;
    state.event.Set(summary);
    metadata->on_device_state(state, 123, 0);
    sink->close_media();
    sink->close_metadata();
    sink->commit_metadata();

    mcap::McapReader reader;
    REQUIRE(reader.open(path.string()).ok());
    std::map<std::string, size_t> counts;
    for (const auto& view : reader.readMessages())
    {
        ++counts[view.channel->topic];
        CHECK(view.message.logTime > 0);
        if (view.channel->topic == "orbbec_ego_device/DeviceState")
        {
            const auto* record = flatbuffers::GetRoot<core::OrbbecEgoDeviceStateRecord>(view.message.data);
            const auto* event = record->data()->event_as_OrbbecEgoCaptureSummary();
            REQUIRE(event);
            CHECK(event->is_final());
            CHECK(event->queue_peak() == 7);
            CHECK(record->data()->observed_at_ns() == record->timestamp()->sample_time_local_common_clock());
        }
        if (view.channel->topic == "orbbec_ego_metadata/ColorLeft" ||
            view.channel->topic == "orbbec_ego_metadata/ColorRight")
        {
            flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize);
            REQUIRE(verifier.VerifyBuffer<core::OrbbecEgoFrameMetadataRecord>());
            const auto* record = flatbuffers::GetRoot<core::OrbbecEgoFrameMetadataRecord>(view.message.data);
            REQUIRE(record->data());
            REQUIRE(record->timestamp());
            CHECK(record->data()->capture_epoch() == 3);
            CHECK(record->timestamp()->sample_time_local_common_clock() == 123 + record->data()->sequence_number());
            CHECK(record->timestamp()->sample_time_raw_device_clock() == 456 + record->data()->sequence_number());
            CHECK(record->timestamp()->available_time_local_common_clock() == static_cast<int64_t>(view.message.logTime));
        }
    }
    CHECK(counts == std::map<std::string, size_t>{ { "orbbec_ego_metadata/ColorLeft", 2 },
                                                   { "orbbec_ego_metadata/ColorRight", 2 },
                                                   { "orbbec_ego_media/ColorLeft", 2 },
                                                   { "orbbec_ego_media/ColorRight", 2 },
                                                   { "orbbec_ego_imu/Accel", 1 },
                                                   { "orbbec_ego_imu/Gyro", 1 },
                                                   { "orbbec_ego_audio/Audio", 1 },
                                                   { "orbbec_ego_media/Audio", 1 },
                                                   { "orbbec_ego_calibration/Calibration", 1 },
                                                   { "orbbec_ego_device/DeviceState", 1 } });
    reader.close();
    std::filesystem::remove(path);
}

#if defined(__linux__)
TEST_CASE("EGO MCAP flush failures prevent final archive promotion", "[ego][mcap][shutdown]")
{
    const auto path = std::filesystem::temp_directory_path() / "ego_full_output_test.mcap";
    const auto partial = std::filesystem::path(path.string() + ".partial");
    std::filesystem::remove(path);
    std::filesystem::remove(partial);
    plugins::ego::CaptureConfig config;
    config.mcap_filename = path.string();
    config.mcap_media_mode = plugins::ego::McapMediaMode::Embedded;
    auto sink = plugins::ego::create_frame_sink({ { core::OrbbecEgoCameraStream_ColorLeft, "" } }, config);
    sink->close_media();
    struct FileLimit
    {
        rlimit previous{};
        using Handler = void (*)(int);
        Handler handler;
        FileLimit()
        {
            REQUIRE(getrlimit(RLIMIT_FSIZE, &previous) == 0);
            const rlimit limit{ 0, previous.rlim_max };
            REQUIRE(setrlimit(RLIMIT_FSIZE, &limit) == 0);
            handler = std::signal(SIGXFSZ, SIG_IGN);
        }
        ~FileLimit()
        {
            setrlimit(RLIMIT_FSIZE, &previous);
            std::signal(SIGXFSZ, handler);
        }
    };
    {
        FileLimit limit;
        CHECK_THROWS(sink->close_metadata());
    }
    CHECK_THROWS(sink->commit_metadata());
    CHECK_FALSE(std::filesystem::exists(path));
    sink.reset();
    std::filesystem::remove(partial);
}
#endif

TEST_CASE("EGO diagnostics off omits schema channel and messages but preserves sensors", "[ego][mcap]")
{
    const auto path = std::filesystem::temp_directory_path() / "ego_diagnostics_off_test.mcap";
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".partial");
    plugins::ego::CaptureConfig config;
    config.mcap_filename = path.string();
    config.mcap_media_mode = plugins::ego::McapMediaMode::Embedded;
    config.record_device_diagnostics = false;
    auto sink = plugins::ego::create_frame_sink({ { core::OrbbecEgoCameraStream_ColorLeft, "" } }, config);
    core::OrbbecEgoDeviceStateT diagnostic;
    diagnostic.device_uid = "fixture";
    diagnostic.observed_at_ns = 100;
    diagnostic.diagnostic_version = 2;
    diagnostic.event.Set(core::OrbbecEgoSdkReportT{});
    sink->metadata_sink()->on_device_state(diagnostic, 100, 0);
    core::OrbbecEgoCalibrationT calibration;
    calibration.device_uid = "test";
    sink->metadata_sink()->on_calibration(calibration, 100, 0);
    sink->close_media();
    sink->close_metadata();
    sink->commit_metadata();
    mcap::McapReader reader;
    REQUIRE(reader.open(path.string()).ok());
    REQUIRE(reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan).ok());
    for (const auto& [id, schema] : reader.schemas())
        CHECK(schema->name != "core.OrbbecEgoDeviceStateRecord");
    CHECK(reader.statistics()->messageCount == 1);
    for (const auto& view : reader.readMessages())
        CHECK(view.channel->topic == "orbbec_ego_calibration/Calibration");
    auto& data = *reader.dataSource();
    mcap::TypedRecordReader records(data, sizeof(mcap::Magic), data.size() - sizeof(mcap::Magic));
    unsigned policies = 0;
    records.onMetadata = [&](const mcap::Metadata& metadata, auto)
    {
        if (metadata.name == "ego.recording")
        {
            ++policies;
            CHECK(metadata.metadata.at("record_device_diagnostics") == "off");
            CHECK(metadata.metadata.at("diagnostic_version") == "2");
        }
    };
    while (records.next())
        REQUIRE(records.status().ok());
    CHECK(policies == 1);
    reader.close();
    std::filesystem::remove(path);
}
