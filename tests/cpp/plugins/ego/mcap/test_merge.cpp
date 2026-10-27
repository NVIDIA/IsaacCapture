// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <ego_mcap/merge.hpp>
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>

#include <array>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <process.h>
#else
#    include <sys/resource.h>

#    include <unistd.h>
#endif

namespace
{
namespace fs = std::filesystem;

struct TempDirectory
{
    fs::path path;
    TempDirectory()
    {
        static std::atomic<unsigned> counter{ 0 };
#ifdef _WIN32
        const auto pid = _getpid();
#else
        const auto pid = getpid();
#endif
        path = fs::temp_directory_path() / ("ego_merge_" + std::to_string(pid) + "_" + std::to_string(counter++));
        REQUIRE(fs::create_directory(path));
    }
    ~TempDirectory()
    {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

void write_fixture(const fs::path& file,
                   const std::string& topic,
                   const std::string& schema_name,
                   const std::vector<uint64_t>& times,
                   bool unknown = false,
                   size_t payload_size = 4,
                   const std::string& diagnostic_policy = {})
{
    mcap::McapWriter writer;
    mcap::McapWriterOptions options("teleop");
    options.compression = mcap::Compression::None;
    options.chunkSize = 1;
    REQUIRE(writer.open(file.string(), options).ok());
    mcap::Schema schema(schema_name, "flatbuffer", "opaque schema fixture");
    writer.addSchema(schema);
    mcap::Channel channel(topic, "flatbuffer", schema.id, { { "source", file.filename().string() } });
    writer.addChannel(channel);
    uint32_t sequence = 0;
    for (const auto time : times)
    {
        std::vector<std::byte> payload(payload_size, std::byte(time));
        mcap::Message message;
        message.channelId = channel.id;
        message.sequence = ++sequence;
        message.logTime = time;
        message.publishTime = time + 5;
        message.dataSize = payload.size();
        message.data = payload.data();
        REQUIRE(writer.write(message).ok());
    }
    mcap::Metadata metadata;
    metadata.name = "fixture";
    metadata.metadata = { { "file", file.filename().string() } };
    REQUIRE(writer.write(metadata).ok());
    if (!diagnostic_policy.empty())
    {
        mcap::Metadata policy;
        policy.name = "ego.recording";
        policy.metadata = { { "record_device_diagnostics", diagnostic_policy }, { "diagnostic_version", "2" } };
        REQUIRE(writer.write(policy).ok());
    }
    const std::array attachment_data{ std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 } };
    mcap::Attachment attachment;
    attachment.logTime = 7;
    attachment.createTime = 6;
    attachment.name = file.filename().string();
    attachment.mediaType = "application/octet-stream";
    attachment.dataSize = attachment_data.size();
    attachment.data = attachment_data.data();
    REQUIRE(writer.write(attachment).ok());
    if (unknown)
    {
        writer.closeLastChunk();
        const std::array<std::byte, 9> record{ std::byte{ 0x80 } };
        writer.dataSink()->write(record.data(), record.size());
    }
    writer.close();
}

void fixture_pair(const TempDirectory& temp, size_t payload_size = 4)
{
    write_fixture(temp.path / "session.mcap", "orbbec_ego_metadata_colorleft/FrameMetadata",
                  "core.OrbbecEgoFrameMetadataRecord", { 30, 10 }, false, payload_size);
    write_fixture(temp.path / "capture.mcap", "orbbec_ego_metadata_colorleft/FrameMetadata",
                  "core.OrbbecEgoFrameMetadataRecord", { 20, 5 }, false, payload_size);
}
} // namespace

TEST_CASE("ego_mcap_merge preserves opaque records and host observations", "[ego][unit]")
{
    TempDirectory temp;
    fixture_pair(temp);
    const auto output = temp.path / "final.mcap";
    plugins::ego::merge_recording(temp.path / "session.mcap", temp.path / "capture.mcap", output);
    REQUIRE(fs::exists(temp.path / "session.mcap"));
    REQUIRE(fs::exists(temp.path / "capture.mcap"));
    REQUIRE_FALSE(fs::exists(output.string() + ".partial"));
    mcap::McapReader reader;
    REQUIRE(reader.open(output.string()).ok());
    REQUIRE(reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan).ok());
    REQUIRE(reader.header()->profile == "teleop");
    REQUIRE(reader.statistics()->messageCount == 4);
    REQUIRE(reader.statistics()->metadataCount == 3);
    REQUIRE(reader.statistics()->attachmentCount == 2);
    std::vector<uint64_t> times;
    std::vector<std::string> topics;
    for (const auto& view : reader.readMessages([](const mcap::Status& status) { REQUIRE(status.ok()); }))
    {
        times.push_back(view.message.logTime);
        topics.push_back(view.channel->topic);
        REQUIRE(view.message.publishTime == view.message.logTime + 5);
        REQUIRE(view.message.sequence == (view.message.logTime == 30 || view.message.logTime == 20 ? 1 : 2));
        REQUIRE(view.message.dataSize == 4);
        for (uint64_t index = 0; index < view.message.dataSize; ++index)
            REQUIRE(view.message.data[index] == std::byte(view.message.logTime));
        REQUIRE(view.channel->metadata.at("source") ==
                (view.channel->topic.starts_with("session_observed/") ? "session.mcap" : "capture.mcap"));
        REQUIRE(view.schema->data.size() == std::string("opaque schema fixture").size());
    }
    REQUIRE(times == std::vector<uint64_t>{ 5, 10, 20, 30 });
    REQUIRE(topics == std::vector<std::string>{ "orbbec_ego_metadata_colorleft/FrameMetadata",
                                                "session_observed/orbbec_ego_metadata_colorleft/FrameMetadata",
                                                "orbbec_ego_metadata_colorleft/FrameMetadata",
                                                "session_observed/orbbec_ego_metadata_colorleft/FrameMetadata" });
    unsigned attachments = 0;
    unsigned metadata = 0;
    auto& data = *reader.dataSource();
    mcap::TypedRecordReader records(data, sizeof(mcap::Magic), data.size() - sizeof(mcap::Magic));
    records.onAttachment = [&](const mcap::Attachment& attachment, auto)
    {
        ++attachments;
        REQUIRE(attachment.createTime == 6);
        REQUIRE(attachment.logTime == 7);
        REQUIRE(attachment.dataSize == 3);
        REQUIRE(attachment.data[2] == std::byte{ 3 });
    };
    records.onMetadata = [&](const mcap::Metadata& record, auto)
    {
        ++metadata;
        if (record.name == "ego.merge")
            REQUIRE(record.metadata.at("recording.library").size() > 0);
        else
            REQUIRE(record.metadata.contains("file"));
    };
    while (records.next())
        REQUIRE(records.status().ok());
    REQUIRE(attachments == 2);
    REQUIRE(metadata == 3);
}

TEST_CASE("ego_mcap_merge preserves ordinary session topics", "[ego][unit]")
{
    TempDirectory temp;
    write_fixture(temp.path / "session.mcap", "head/Head", "core.HeadPoseRecord", { 10 });
    write_fixture(
        temp.path / "capture.mcap", "orbbec_ego_media/ColorLeft", "core.OrbbecEgoEncodedVideoFrameRecord", { 5 });
    plugins::ego::merge_recording(temp.path / "session.mcap", temp.path / "capture.mcap", temp.path / "final.mcap");
    mcap::McapReader reader;
    REQUIRE(reader.open((temp.path / "final.mcap").string()).ok());
    REQUIRE(reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan).ok());
    bool head = false;
    for (const auto& [id, channel] : reader.channels())
        head |= channel->topic == "head/Head";
    REQUIRE(head);
}

TEST_CASE("ego_mcap_merge rejects ambiguous channels and unknown records", "[ego][unit]")
{
    TempDirectory temp;
    write_fixture(temp.path / "session.mcap", "shared", "core.OrbbecEgoFrameMetadataRecord", { 10 });
    SECTION("different schema")
    {
        write_fixture(temp.path / "capture.mcap", "shared", "core.OrbbecEgoImuBatchRecord", { 5 });
    }
    SECTION("reserved topic prefix")
    {
        write_fixture(temp.path / "capture.mcap", "session_observed/existing", "core.OrbbecEgoImuBatchRecord", { 5 });
    }
    SECTION("unknown semantic record")
    {
        write_fixture(temp.path / "capture.mcap", "camera", "core.OrbbecEgoImuBatchRecord", { 5 }, true);
    }
    REQUIRE_THROWS(plugins::ego::merge_recording(
        temp.path / "session.mcap", temp.path / "capture.mcap", temp.path / "final.mcap"));
    REQUIRE_FALSE(fs::exists(temp.path / "final.mcap"));
    REQUIRE(fs::exists(temp.path / "capture.mcap"));
}

TEST_CASE("ego_mcap_merge rejects truncated input and existing output", "[ego][unit]")
{
    TempDirectory temp;
    fixture_pair(temp);
    SECTION("truncated footer")
    {
        const auto path = temp.path / "capture.mcap";
        fs::resize_file(path, fs::file_size(path) - 6);
    }
    SECTION("existing final")
    {
        std::ofstream(temp.path / "final.mcap") << "keep";
    }
    SECTION("existing partial")
    {
        std::ofstream(temp.path / "final.mcap.partial") << "keep";
    }
    REQUIRE_THROWS(plugins::ego::merge_recording(
        temp.path / "session.mcap", temp.path / "capture.mcap", temp.path / "final.mcap"));
    REQUIRE(fs::exists(temp.path / "session.mcap"));
    REQUIRE(fs::exists(temp.path / "capture.mcap"));
}

#ifndef _WIN32
TEST_CASE("ego_mcap_merge retains partial output on write failure", "[ego][unit]")
{
    TempDirectory temp;
    fixture_pair(temp, 16384);
    struct FileLimit
    {
        rlimit previous{};
        using Handler = void (*)(int);
        Handler handler;
        FileLimit()
        {
            REQUIRE(getrlimit(RLIMIT_FSIZE, &previous) == 0);
            handler = std::signal(SIGXFSZ, SIG_IGN);
            const rlimit limit{ 1024, previous.rlim_max };
            REQUIRE(setrlimit(RLIMIT_FSIZE, &limit) == 0);
        }
        ~FileLimit()
        {
            setrlimit(RLIMIT_FSIZE, &previous);
            std::signal(SIGXFSZ, handler);
        }
    };
    {
        FileLimit limit;
        REQUIRE_THROWS(plugins::ego::merge_recording(
            temp.path / "session.mcap", temp.path / "capture.mcap", temp.path / "final.mcap"));
    }
    REQUIRE_FALSE(fs::exists(temp.path / "final.mcap"));
    REQUIRE(fs::exists(temp.path / "final.mcap.partial"));
    REQUIRE(fs::exists(temp.path / "session.mcap"));
    REQUIRE(fs::exists(temp.path / "capture.mcap"));
}
#endif

TEST_CASE("ego_mcap_merge enforces diagnostics off across both inputs by schema", "[ego][unit]")
{
    TempDirectory temp;
    const auto parent = temp.path / "session.mcap";
    const auto fragment = temp.path / "capture.mcap";
    const auto output = temp.path / "final.mcap";
    SECTION("parent contains renamed diagnostics")
    {
        write_fixture(parent, "custom/diagnostic", "core.OrbbecEgoDeviceStateRecord", { 1 });
        write_fixture(fragment, "video", "core.OrbbecEgoFrameMetadataRecord", { 2 }, false, 4, "off");
    }
    SECTION("fragment contains diagnostics despite policy")
    {
        write_fixture(parent, "head", "core.HeadPoseRecord", { 1 });
        write_fixture(fragment, "custom/diagnostic", "core.OrbbecEgoDeviceStateRecord", { 2 }, false, 4, "off");
    }
    SECTION("parent registers an empty diagnostic channel")
    {
        write_fixture(parent, "empty/diagnostic", "core.OrbbecEgoDeviceStateRecord", {});
        write_fixture(fragment, "video", "core.OrbbecEgoFrameMetadataRecord", { 2 }, false, 4, "off");
    }
    SECTION("fragment registers an empty diagnostic channel")
    {
        write_fixture(parent, "head", "core.HeadPoseRecord", { 1 });
        write_fixture(fragment, "empty/diagnostic", "core.OrbbecEgoDeviceStateRecord", {}, false, 4, "off");
    }
    SECTION("parent declares a conflicting policy")
    {
        write_fixture(parent, "head", "core.HeadPoseRecord", { 1 }, false, 4, "on");
        write_fixture(fragment, "video", "core.OrbbecEgoFrameMetadataRecord", { 2 }, false, 4, "off");
    }
    REQUIRE_THROWS(plugins::ego::merge_recording(parent, fragment, output));
    CHECK(fs::exists(parent));
    CHECK(fs::exists(fragment));
    CHECK_FALSE(fs::exists(output));
}

TEST_CASE("ego_mcap_merge preserves off policy and every other record", "[ego][unit]")
{
    TempDirectory temp;
    write_fixture(temp.path / "session.mcap", "head", "core.HeadPoseRecord", { 1 });
    write_fixture(temp.path / "capture.mcap", "video", "core.OrbbecEgoFrameMetadataRecord", { 2 }, false, 4, "off");
    plugins::ego::merge_recording(temp.path / "session.mcap", temp.path / "capture.mcap", temp.path / "final.mcap");
    mcap::McapReader reader;
    REQUIRE(reader.open((temp.path / "final.mcap").string()).ok());
    REQUIRE(reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan).ok());
    CHECK(reader.statistics()->messageCount == 2);
    CHECK(reader.statistics()->metadataCount == 4);
    CHECK(reader.statistics()->attachmentCount == 2);
    auto& data = *reader.dataSource();
    mcap::TypedRecordReader records(data, sizeof(mcap::Magic), data.size() - sizeof(mcap::Magic));
    unsigned policy_count = 0;
    records.onMetadata = [&](const mcap::Metadata& metadata, auto)
    {
        if (metadata.name == "ego.recording")
        {
            ++policy_count;
            CHECK(metadata.metadata.at("record_device_diagnostics") == "off");
        }
    };
    while (records.next())
        REQUIRE(records.status().ok());
    CHECK(policy_count == 1);
}
