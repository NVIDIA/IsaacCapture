// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/ego_audio_bfbs_generated.h>
#include <schema/ego_audio_generated.h>
#include <schema/ego_camera_bfbs_generated.h>
#include <schema/ego_camera_generated.h>
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

void write_fixture(const fs::path& path, bool wrong_video_stream = false, bool wrong_pcm_size = false)
{
    mcap::McapWriter writer;
    mcap::McapWriterOptions options("ego");
    options.compression = mcap::Compression::None;
    REQUIRE(writer.open(path.string(), options).ok());
    core::McapTrackerChannels<core::EgoEncodedVideoFrameRecord> video(writer, "ego_media", { "ColorLeft", "ColorRight" });
    core::McapTrackerChannels<core::EgoPcmAudioChunkRecord> audio(writer, "ego_media", { "Audio" });
    for (size_t index = 0; index < 2; ++index)
    {
        for (uint64_t sequence = 0; sequence < 2; ++sequence)
        {
            core::EgoEncodedVideoFrameT frame;
            frame.stream = static_cast<core::EgoCameraStream>(wrong_video_stream ? 1 - index : index);
            frame.sequence_number = sequence;
            frame.width = 1600;
            frame.height = 1300;
            frame.fps = 30;
            frame.encoded_data = { 0xff, 0xd8, static_cast<uint8_t>(index), 0xff, 0xd9 };
            video.write(index, core::pack_record<core::EgoEncodedVideoFrameRecord>(
                                   &frame, core::DeviceDataTimestamp(100 + sequence, 90 + sequence, 80 + sequence)));
        }
    }
    core::EgoPcmAudioChunkT pcm;
    pcm.sample_rate_hz = 48000;
    pcm.channel_count = 1;
    pcm.bits_per_sample = 16;
    pcm.sample_count = wrong_pcm_size ? 3 : 2;
    pcm.pcm_data = { 1, 2, 3, 4 };
    audio.write(0, core::pack_record<core::EgoPcmAudioChunkRecord>(&pcm, core::DeviceDataTimestamp(101, 91, 81)));
    writer.close();
}
} // namespace

TEST_CASE("ego_media_export preserves encoded bytes and PCM WAV", "[ego][unit]")
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
    const auto wav = read_bytes(output / "Audio.wav");
    REQUIRE(wav.size() == 48);
    CHECK(std::string(wav.begin(), wav.begin() + 4) == "RIFF");
    CHECK(wav[4] == 40);
    CHECK(wav[22] == 1);
    CHECK(wav[24] == 0x80);
    CHECK(wav[25] == 0xbb);
    CHECK(wav[34] == 16);
    CHECK(wav[40] == 4);
    CHECK(std::vector<uint8_t>(wav.begin() + 44, wav.end()) == std::vector<uint8_t>{ 1, 2, 3, 4 });
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
    SECTION("PCM size does not match sample count")
    {
        write_fixture(source, false, true);
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
        mcap::Schema schema("core.EgoEncodedVideoFrameRecord", "jsonschema", "{}");
        writer.addSchema(schema);
        mcap::Channel channel("ego_media/ColorLeft", "flatbuffer", schema.id);
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
