// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The metadata half of a generated tracker: the reader's helpers, how the metadata is recorded,
// and how replay reads it back. The tracker is generated from trackers.toml next to this file.

#include "mcap_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <live_trackers/schema_metadata_reader.hpp>
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>
#include <schema/pedals_generated.h>
#include <schema/test_tracker_metadata_generated.h>
#include <schema/tracker_metadata.hpp>

#include <XR_NVX1_tensor_data.h>
#include <cstdint>
#include <live_metadata_test_tracker_impl.hpp>
#include <memory>
#include <replay_metadata_test_tracker_impl.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using namespace mcap_test;

constexpr std::string_view BASE_NAME = "pedals";

core::Serialized<core::TestTrackerMetadata> make_metadata(const std::string& label)
{
    core::TestTrackerMetadataT native;
    native.label = label;
    native.sizes = { 3, 4 };
    return core::pack<core::TestTrackerMetadata>(native);
}

core::Serialized<core::Generic3AxisPedalOutputRecord> make_record(float left_pedal, int64_t time_ns)
{
    core::Generic3AxisPedalOutputT native;
    native.left_pedal = left_pedal;
    return core::pack_record<core::Generic3AxisPedalOutputRecord>(
        &native, core::DeviceDataTimestamp(time_ns, time_ns, time_ns));
}

std::unique_ptr<mcap::McapWriter> open_writer(const std::string& path)
{
    auto writer = std::make_unique<mcap::McapWriter>();
    mcap::McapWriterOptions options("teleop-test");
    options.compression = mcap::Compression::None;
    REQUIRE(writer->open(path, options).ok());
    return writer;
}

//! Writes what a live session of the tracker records: the metadata, when there is one, then one
//! sample on the tracked channel replay reads.
void write_recording(const std::string& path, const core::Serialized<core::TestTrackerMetadata>& metadata)
{
    auto writer = open_writer(path);
    auto channels = core::LiveMetadataTestTrackerImpl::create_mcap_channels(*writer, BASE_NAME);
    if (metadata)
    {
        channels->metadata.write(0, metadata, 5);
    }
    channels->data.write(1, make_record(0.25f, 10));
    writer->close();
}

//! Builds the replay impl the way the replay factory does.
std::unique_ptr<core::ReplayMetadataTestTrackerImpl> open_replay(const std::string& path)
{
    const auto recorded = recorded_schemas(path);
    auto metadata = core::ReplayMetadataTestTrackerImpl::read_metadata(open_reader(path), BASE_NAME, recorded);
    return std::make_unique<core::ReplayMetadataTestTrackerImpl>(
        open_reader(path), BASE_NAME, recorded, std::move(metadata));
}

} // namespace

TEST_CASE("The metadata of a collection is read from <collection>/metadata", "[unit][tracker_metadata]")
{
    const auto config = core::tracker_metadata_reader_config("tactile");
    CHECK(config.collection_id == core::tracker_metadata_collection_id("tactile"));
    CHECK(config.collection_id == "tactile/metadata");
    CHECK(config.tensor_identifier == core::TRACKER_METADATA_TENSOR_IDENTIFIER);
    CHECK(config.max_flatbuffer_size == core::TRACKER_METADATA_MAX_SIZE);

    const std::string longest(XR_MAX_TENSOR_IDENTIFIER_SIZE - 1 - std::string_view("/metadata").size(), 'c');
    CHECK_NOTHROW(core::tracker_metadata_reader_config(longest));
    CHECK_THROWS_AS(core::tracker_metadata_reader_config(longest + "c"), std::invalid_argument);
}

TEST_CASE("Metadata samples compare equal up to their zero padding", "[unit][tracker_metadata]")
{
    const std::vector<uint8_t> table{ 4, 0, 7, 9 };
    std::vector<uint8_t> padded = table;
    padded.resize(16, 0);

    CHECK(core::same_tracker_metadata(table, table));
    CHECK(core::same_tracker_metadata(table, padded));
    CHECK(core::same_tracker_metadata(padded, table));

    std::vector<uint8_t> other_padding = padded;
    other_padding.back() = 1;
    CHECK_FALSE(core::same_tracker_metadata(table, other_padding));

    std::vector<uint8_t> other_table = table;
    other_table[2] = 8;
    CHECK_FALSE(core::same_tracker_metadata(table, other_table));
}

TEST_CASE("A metadata table is recorded at the log time it is given", "[unit][tracker_metadata]")
{
    const auto path = temp_mcap_path("test_tracker_metadata");
    const TempFileCleanup cleanup(path);
    write_recording(path, make_metadata("glove"));

    auto reader = open_reader(path);
    const auto on_problem = [](const mcap::Status& status) { FAIL(status.message); };
    int metadata_messages = 0;
    for (const auto& view : reader->readMessages(on_problem))
    {
        if (view.channel->topic != "pedals/metadata_test_metadata")
        {
            continue;
        }
        ++metadata_messages;
        CHECK(view.schema->name == core::TestTrackerMetadata::GetFullyQualifiedName());
        CHECK(view.message.logTime == 5);
    }
    CHECK(metadata_messages == 1);
}

TEST_CASE("Replay publishes the data with the metadata it was recorded with", "[unit][tracker_metadata]")
{
    const auto path = temp_mcap_path("test_tracker_metadata");
    const TempFileCleanup cleanup(path);
    write_recording(path, make_metadata("glove"));

    auto replay = open_replay(path);
    REQUIRE(replay->get_metadata());
    CHECK(replay->get_metadata()->label()->str() == "glove");

    replay->update(0);
    REQUIRE(replay->get_data());
    CHECK(replay->get_data()->left_pedal() == 0.25f);
}

TEST_CASE("Replay withholds data recorded without metadata", "[unit][tracker_metadata]")
{
    const auto path = temp_mcap_path("test_tracker_metadata");
    const TempFileCleanup cleanup(path);
    write_recording(path, core::Serialized<core::TestTrackerMetadata>());

    auto replay = open_replay(path);
    CHECK_FALSE(replay->get_metadata());

    replay->update(0);
    CHECK_FALSE(replay->get_data());
}
