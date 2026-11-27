// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for McapTrackerChannels<RecordT>.

#define MCAP_IMPLEMENTATION

#include "mcap_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <mcap/recording_traits.hpp>
#include <mcap/writer.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{

using namespace mcap_test;


} // namespace

// =============================================================================
// McapTrackerChannels - typed write + readback
// =============================================================================

TEST_CASE("McapTrackerChannels: typed write produces readable MCAP with correct record content",
          "[mcap][tracker_channels]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT head_data;
    head_data.is_valid = true;
    head_data.pose =
        std::make_shared<core::Pose>(core::Point(1.0f, 2.0f, 3.0f), core::Quaternion(0.0f, 0.0f, 0.707f, 0.707f));

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "head" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&head_data, core::DeviceDataTimestamp(1000000, 1000000, 42)));
        writer->close();
    }

    mcap::McapReader reader;
    REQUIRE(reader.open(path).ok());

    size_t msg_count = 0;
    for (const auto& view : reader.readMessages())
    {
        CHECK(view.channel->topic == "tracking/head");
        CHECK(view.schema->name == core::HeadPoseRecord::GetFullyQualifiedName());
        CHECK(view.message.logTime == 1000000);

        auto record = flatbuffers::GetRoot<core::HeadPoseRecord>(view.message.data);
        REQUIRE(record != nullptr);
        REQUIRE(record->timestamp() != nullptr);
        CHECK(record->timestamp()->sample_time_raw_device_clock() == 42);
        REQUIRE(record->data() != nullptr);
        CHECK(record->data()->is_valid() == true);

        REQUIRE(record->data()->pose() != nullptr);
        CHECK(record->data()->pose()->position().x() == 1.0f);
        CHECK(record->data()->pose()->position().y() == 2.0f);
        CHECK(record->data()->pose()->position().z() == 3.0f);
        CHECK(record->data()->pose()->orientation().x() == 0.0f);
        CHECK(record->data()->pose()->orientation().y() == 0.0f);
        CHECK(record->data()->pose()->orientation().z() == 0.707f);
        CHECK(record->data()->pose()->orientation().w() == 0.707f);

        msg_count++;
    }
    CHECK(msg_count == 1);
    reader.close();
}

TEST_CASE("McapTrackerChannels: an encoded record can be written to a second channel", "[mcap][tracker_channels]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT head_data;
    head_data.is_valid = true;
    head_data.pose =
        std::make_shared<core::Pose>(core::Point(7.0f, 8.0f, 9.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "head", "head_tracked" });

        // What a generated pull tracker does: every sample to the per-sample channel, the
        // last one over again to the tracked channel -- without encoding it twice.
        const auto record = core::pack_record<core::HeadPoseRecord>(&head_data, core::DeviceDataTimestamp(11, 11, 22));
        ch.write(0, record);
        ch.write(1, record);
        writer->close();
    }

    mcap::McapReader reader;
    REQUIRE(reader.open(path).ok());

    std::vector<std::vector<uint8_t>> payloads;
    std::vector<std::string> topics;
    for (const auto& view : reader.readMessages())
    {
        const auto* bytes = reinterpret_cast<const uint8_t*>(view.message.data);
        payloads.emplace_back(bytes, bytes + view.message.dataSize);
        topics.push_back(view.channel->topic);
    }
    reader.close();

    REQUIRE(payloads.size() == 2);
    CHECK(topics[0] == "tracking/head");
    CHECK(topics[1] == "tracking/head_tracked");

    // Byte-identical: that is what makes reusing the encode legal rather than a shortcut.
    CHECK(payloads[0] == payloads[1]);

    auto replayed = flatbuffers::GetRoot<core::HeadPoseRecord>(payloads[1].data());
    REQUIRE(replayed->data() != nullptr);
    CHECK(replayed->data()->pose()->position().x() == 7.0f);
    CHECK(replayed->timestamp()->sample_time_raw_device_clock() == 22);
}

TEST_CASE("publish_and_record encodes the payload alone when recording is off", "[mcap][tracker_channels]")
{
    core::HeadPoseT native;
    native.is_valid = true;
    native.pose = std::make_shared<core::Pose>(core::Point(1.0f, 2.0f, 3.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));

    // Null channels is how an impl spells "recording disabled": there is no Record to
    // write, so the payload is encoded on its own and published from that buffer.
    const auto published =
        core::publish_and_record<core::HeadPoseRecord>(nullptr, 0, core::DeviceDataTimestamp(1, 1, 1), &native);
    REQUIRE(published);
    CHECK(published->is_valid() == true);
    CHECK(published->pose()->position().y() == 2.0f);

    // An inactive device lands on an empty handle whether or not anything is recording.
    CHECK_FALSE(core::publish_and_record<core::HeadPoseRecord>(nullptr, 0, core::DeviceDataTimestamp(1, 1, 1), nullptr));

    const std::optional<core::HeadPoseT> absent;
    CHECK_FALSE(core::publish_and_record<core::HeadPoseRecord>(
        nullptr, 0, core::DeviceDataTimestamp(1, 1, 1), core::value_ptr(absent)));
}

TEST_CASE("McapTrackerChannels: write returns the record it wrote", "[mcap][tracker_channels]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    auto head_data = std::make_shared<core::HeadPoseT>();
    head_data->is_valid = true;
    head_data->pose =
        std::make_shared<core::Pose>(core::Point(4.0f, 5.0f, 6.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));

    core::Serialized<core::HeadPose> published;
    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "head" });

        const auto record = core::pack_record<core::HeadPoseRecord>(head_data.get(), core::DeviceDataTimestamp(7, 7, 9));
        ch.write(0, record);
        REQUIRE(record);
        CHECK(record->timestamp()->sample_time_raw_device_clock() == 9);

        // What a live impl does with it: publish a view into the recorded bytes instead of
        // encoding the payload a second time. The record handle dies here, so the narrowed
        // one is all that keeps the buffer alive.
        published = record.narrow(record->data());
    }

    REQUIRE(published);
    CHECK(published->is_valid() == true);
    CHECK(published->pose()->position().x() == 4.0f);
    CHECK(published->pose()->position().z() == 6.0f);

    // The published handle and the recorded message must be the same values -- one encode
    // served both.
    mcap::McapReader reader;
    REQUIRE(reader.open(path).ok());
    size_t msg_count = 0;
    for (const auto& view : reader.readMessages())
    {
        auto record = flatbuffers::GetRoot<core::HeadPoseRecord>(view.message.data);
        REQUIRE(record->data() != nullptr);
        CHECK(record->data()->is_valid() == published->is_valid());
        CHECK(record->data()->pose()->position().x() == published->pose()->position().x());
        CHECK(record->data()->pose()->position().z() == published->pose()->position().z());
        msg_count++;
    }
    CHECK(msg_count == 1);
    reader.close();
}

TEST_CASE("pack_record on an absent payload yields an empty payload handle", "[mcap][tracker_channels]")
{
    // An inactive device still writes a record so the frame is marked, but there is no
    // payload to publish from it.
    const auto record = core::pack_record<core::HeadPoseRecord>(nullptr, core::DeviceDataTimestamp(1, 1, 2));

    REQUIRE(record);
    CHECK(record->timestamp()->sample_time_raw_device_clock() == 2);
    CHECK(record->data() == nullptr);
    CHECK_FALSE(record.narrow(record->data()));
}

TEST_CASE("McapTrackerChannels: null data writes record with timestamp only", "[mcap][tracker_channels]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "head" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(nullptr, core::DeviceDataTimestamp(500, 500, 10)));
        writer->close();
    }

    mcap::McapReader reader;
    REQUIRE(reader.open(path).ok());

    size_t msg_count = 0;
    for (const auto& view : reader.readMessages())
    {
        auto record = flatbuffers::GetRoot<core::HeadPoseRecord>(view.message.data);
        REQUIRE(record != nullptr);
        REQUIRE(record->timestamp() != nullptr);
        CHECK(record->timestamp()->sample_time_raw_device_clock() == 10);
        CHECK(record->data() == nullptr);

        msg_count++;
    }
    CHECK(msg_count == 1);
    reader.close();
}

TEST_CASE("McapTrackerChannels: multi-channel write routes to correct topics", "[mcap][tracker_channels]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT data;

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "hands", { "left", "right" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(100, 100, 1)));
        ch.write(1, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(200, 200, 2)));
        writer->close();
    }

    mcap::McapReader reader;
    REQUIRE(reader.open(path).ok());

    std::vector<std::string> topics;
    for (const auto& view : reader.readMessages())
    {
        topics.push_back(view.channel->topic);
    }

    REQUIRE(topics.size() == 2);
    CHECK(topics[0] == "hands/left");
    CHECK(topics[1] == "hands/right");
    reader.close();
}

TEST_CASE("McapTrackerChannels: out-of-range channel_index throws", "[mcap][tracker_channels]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT data;

    auto writer = open_writer(path);
    HeadChannels ch(*writer, "test", { "only" });
    CHECK_THROWS_AS(ch.write(99, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(100, 100, 1))),
                    std::out_of_range);
    writer->close();
}

TEST_CASE("McapTrackerChannels: sequence numbers increment across writes", "[mcap][tracker_channels]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT data;

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "seq", { "ch" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(100, 100, 1)));
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(200, 200, 2)));
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(300, 300, 3)));
        writer->close();
    }

    mcap::McapReader reader;
    REQUIRE(reader.open(path).ok());

    std::vector<uint32_t> sequences;
    for (const auto& view : reader.readMessages())
    {
        sequences.push_back(view.message.sequence);
    }

    REQUIRE(sequences.size() == 3);
    CHECK(sequences[0] == 0);
    CHECK(sequences[1] == 1);
    CHECK(sequences[2] == 2);
    reader.close();
}

TEST_CASE("McapTrackerChannels: multiple same-type channel instances share one writer", "[mcap][tracker_channels]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT data;

    {
        auto writer = open_writer(path);
        HeadChannels head_ch(*writer, "head", { "pose" });
        HeadChannels ctrl_ch(*writer, "ctrl", { "left", "right" });

        head_ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(100, 100, 1)));
        ctrl_ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(200, 200, 2)));
        ctrl_ch.write(1, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(300, 300, 3)));
        writer->close();
    }

    mcap::McapReader reader;
    REQUIRE(reader.open(path).ok());

    std::vector<std::string> topics;
    for (const auto& view : reader.readMessages())
    {
        topics.push_back(view.channel->topic);
    }

    REQUIRE(topics.size() == 3);
    CHECK(topics[0] == "head/pose");
    CHECK(topics[1] == "ctrl/left");
    CHECK(topics[2] == "ctrl/right");
    reader.close();
}

// =============================================================================
// McapTrackerViewers - typed read from specific channels by index
// =============================================================================

TEST_CASE("McapTrackerViewers: reads records from a single channel", "[mcap][tracker_viewers]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT head_data;
    head_data.is_valid = true;
    head_data.pose =
        std::make_shared<core::Pose>(core::Point(1.0f, 2.0f, 3.0f), core::Quaternion(0.0f, 0.0f, 0.707f, 0.707f));

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "head" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&head_data, core::DeviceDataTimestamp(1000000, 1000000, 42)));
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&head_data, core::DeviceDataTimestamp(2000000, 2000000, 84)));
        writer->close();
    }

    const auto recorded = recorded_schemas(path);
    HeadViewers viewers(open_reader(path), "tracking", { "head" }, recorded);

    auto record1 = viewers.read(0);
    REQUIRE(record1);
    REQUIRE(record1->data() != nullptr);
    CHECK(record1->data()->is_valid() == true);
    REQUIRE(record1->data()->pose() != nullptr);
    CHECK(record1->data()->pose()->position().x() == 1.0f);
    REQUIRE(record1->timestamp() != nullptr);
    CHECK(record1->timestamp()->sample_time_raw_device_clock() == 42);

    auto record2 = viewers.read(0);
    REQUIRE(record2);
    REQUIRE(record2->data() != nullptr);
    REQUIRE(record2->timestamp() != nullptr);
    CHECK(record2->timestamp()->sample_time_raw_device_clock() == 84);

    CHECK_FALSE(viewers.read(0));
}

TEST_CASE("McapTrackerViewers: multi-channel reads filter by index", "[mcap][tracker_viewers]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT data;

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "left", "right" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(100, 100, 1)));
        ch.write(1, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(200, 200, 2)));
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(300, 300, 3)));
        ch.write(1, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(400, 400, 4)));
        writer->close();
    }

    const auto recorded = recorded_schemas(path);
    HeadViewers viewers(open_reader(path), "tracking", { "left", "right" }, recorded);

    auto left1 = viewers.read(0);
    REQUIRE(left1);

    auto right1 = viewers.read(1);
    REQUIRE(right1);

    auto left2 = viewers.read(0);
    REQUIRE(left2);

    auto right2 = viewers.read(1);
    REQUIRE(right2);

    CHECK_FALSE(viewers.read(0));
    CHECK_FALSE(viewers.read(1));
}

TEST_CASE("McapTrackerViewers: read subset of written channels", "[mcap][tracker_viewers]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT data;
    data.is_valid = true;

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "left", "right" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(100, 100, 1)));
        ch.write(1, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(200, 200, 2)));
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(300, 300, 3)));
        ch.write(1, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(400, 400, 4)));
        writer->close();
    }

    const auto recorded = recorded_schemas(path);
    HeadViewers viewers(open_reader(path), "tracking", { "right" }, recorded);

    auto r1 = viewers.read(0);
    REQUIRE(r1);
    REQUIRE(r1->data() != nullptr);
    CHECK(r1->data()->is_valid() == true);

    auto r2 = viewers.read(0);
    REQUIRE(r2);
    REQUIRE(r2->data() != nullptr);

    CHECK_FALSE(viewers.read(0));
}

TEST_CASE("McapTrackerViewers: out-of-range channel_index throws", "[mcap][tracker_viewers]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    core::HeadPoseT data;

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "head" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(100, 100, 1)));
        writer->close();
    }

    const auto recorded = recorded_schemas(path);
    HeadViewers viewers(open_reader(path), "tracking", { "head" }, recorded);
    CHECK_THROWS_AS(viewers.read(99), std::out_of_range);
}

TEST_CASE("McapTrackerViewers: handles null data records", "[mcap][tracker_viewers]")
{
    mcap_test::TemporaryMcapFile recording;
    const auto& path = recording.path();

    {
        auto writer = open_writer(path);
        HeadChannels ch(*writer, "tracking", { "head" });
        ch.write(0, core::pack_record<core::HeadPoseRecord>(nullptr, core::DeviceDataTimestamp(500, 500, 10)));
        writer->close();
    }

    const auto recorded = recorded_schemas(path);
    HeadViewers viewers(open_reader(path), "tracking", { "head" }, recorded);

    auto record = viewers.read(0);
    REQUIRE(record);
    CHECK(record->data() == nullptr);
    REQUIRE(record->timestamp() != nullptr);
    CHECK(record->timestamp()->sample_time_raw_device_clock() == 10);

    CHECK_FALSE(viewers.read(0));
}
