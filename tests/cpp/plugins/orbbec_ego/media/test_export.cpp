// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/orbbec_ego_camera_bfbs_generated.h>
#include <schema/orbbec_ego_camera_generated.h>
#include <sys/wait.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{
namespace fs = std::filesystem;

struct TemporaryDirectory
{
    fs::path path;
    TemporaryDirectory()
    {
        static std::atomic<unsigned> counter{ 0 };
        path =
            fs::temp_directory_path() / ("ego_media_test_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
        fs::create_directory(path);
    }
    ~TemporaryDirectory()
    {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

std::vector<uint8_t> read_bytes(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

int run_export(const fs::path& source, const fs::path& output)
{
    const pid_t process = fork();
    REQUIRE(process >= 0);
    if (process == 0)
    {
        execl(EGO_EXPORTER_PATH, EGO_EXPORTER_PATH, source.c_str(), output.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    REQUIRE(waitpid(process, &status, 0) == process);
    REQUIRE(WIFEXITED(status));
    return WEXITSTATUS(status);
}

void write_fixture(const fs::path& path, bool wrong_video_stream = false)
{
    mcap::McapWriter writer;
    mcap::McapWriterOptions options("ego");
    options.compression = mcap::Compression::None;
    REQUIRE(writer.open(path.string(), options).ok());
    core::McapTrackerChannels<core::OrbbecEgoEncodedVideoFrameRecord> video(
        writer, "orbbec_ego_media", { "ColorLeft", "ColorRight" });
    for (size_t index = 0; index < 2; ++index)
    {
        for (uint64_t sequence = 0; sequence < 2; ++sequence)
        {
            core::OrbbecEgoEncodedVideoFrameT frame;
            frame.stream = static_cast<core::OrbbecEgoCameraStream>(wrong_video_stream ? 1 - index : index);
            frame.sequence_number = sequence;
            frame.width = 1600;
            frame.height = 1300;
            frame.fps = 30;
            frame.encoded_data = { 0xff, 0xd8, static_cast<uint8_t>(index), 0xff, 0xd9 };
            video.write(index, core::pack_record<core::OrbbecEgoEncodedVideoFrameRecord>(
                                   &frame, core::DeviceDataTimestamp(100 + sequence, 90 + sequence, 80 + sequence)));
        }
    }
    writer.close();
}
} // namespace

TEST_CASE("ego_media_export preserves encoded bytes", "[ego][unit]")
{
    TemporaryDirectory directory;
    const auto source = directory.path / "source.mcap";
    const auto output = directory.path / "export";
    write_fixture(source);
    REQUIRE(run_export(source, output) == 0);
    CHECK(read_bytes(output / "ColorLeft.mjpg") ==
          std::vector<uint8_t>{ 0xff, 0xd8, 0, 0xff, 0xd9, 0xff, 0xd8, 0, 0xff, 0xd9 });
    CHECK(read_bytes(output / "ColorRight.mjpg") ==
          std::vector<uint8_t>{ 0xff, 0xd8, 1, 0xff, 0xd9, 0xff, 0xd8, 1, 0xff, 0xd9 });
}

TEST_CASE("ego_media_export rejects mismatched media and incomplete input", "[ego][unit]")
{
    TemporaryDirectory directory;
    const auto source = directory.path / "source.mcap";
    const auto output = directory.path / "export";
    SECTION("video topic does not match payload")
    {
        write_fixture(source, true);
    }
    SECTION("footer is truncated")
    {
        write_fixture(source);
        fs::resize_file(source, fs::file_size(source) - 6);
    }
    SECTION("video schema has the wrong encoding")
    {
        mcap::McapWriter writer;
        REQUIRE(writer.open(source.string(), mcap::McapWriterOptions("ego")).ok());
        mcap::Schema schema("core.OrbbecEgoEncodedVideoFrameRecord", "jsonschema", "{}");
        writer.addSchema(schema);
        mcap::Channel channel("orbbec_ego_media/ColorLeft", "flatbuffer", schema.id);
        writer.addChannel(channel);
        writer.close();
    }
    CHECK(run_export(source, output) != 0);
    CHECK(fs::exists(source));
}

#if defined(__linux__)
TEST_CASE("ego_media_export reports final output flush failure", "[ego][unit]")
{
    TemporaryDirectory directory;
    const auto source = directory.path / "source.mcap";
    const auto output = directory.path / "export";
    write_fixture(source);
    fs::create_directory(output);
    fs::create_symlink("/dev/full", output / "ColorLeft.mjpg");
    CHECK(run_export(source, output) != 0);
    CHECK(fs::exists(source));
}
#endif
