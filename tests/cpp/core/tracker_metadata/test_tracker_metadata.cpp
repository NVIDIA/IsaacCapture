// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The metadata half of a generated tracker: the reader's helpers, when live publishes and records
// the data, and how replay reads the recording back. The tracker is generated from trackers.toml
// next to this file; on timespec platforms, a fake tensor runtime stands in for OpenXR.

#if defined(XR_USE_TIMESPEC)
#    include "fake_tensor_runtime.hpp"
#endif
#include "live_metadata_test_tracker_impl.hpp"
#include "mcap_test_support.hpp"
#include "replay_metadata_test_tracker_impl.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <live_trackers/schema_metadata_reader.hpp>
#include <mcap/reader.hpp>
#include <mcap/recording_traits.hpp>
#include <mcap/writer.hpp>
#include <schema/pedals_generated.h>
#include <schema/test_tracker_metadata_generated.h>
#include <schema/tracker_metadata.hpp>

#include <XR_NVX1_tensor_data.h>
#include <cstdint>
#include <memory>
#include <optional>
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

#if defined(XR_USE_TIMESPEC)

using tracker_metadata_test::FakeTensorRuntime;

//! The left pedal of what `get_data()` publishes, or nullopt when it publishes nothing.
std::optional<float> published(const core::Serialized<core::Generic3AxisPedalOutput>& data)
{
    return data ? std::optional<float>(data->left_pedal()) : std::nullopt;
}

//! Every message recorded on `sub_channel`, in order.
template <typename RootT>
std::vector<core::Serialized<RootT>> read_channel(const std::string& path, std::string_view sub_channel)
{
    const auto recorded = recorded_schemas(path);
    core::McapTrackerViewers<RootT> viewers(open_reader(path), BASE_NAME, { std::string(sub_channel) }, recorded);
    std::vector<core::Serialized<RootT>> messages;
    while (auto message = viewers.read(0))
    {
        messages.push_back(std::move(message));
    }
    return messages;
}

std::vector<float> recorded_left_pedals(const std::string& path, std::string_view sub_channel)
{
    std::vector<float> values;
    for (const auto& record : read_channel<core::Generic3AxisPedalOutputRecord>(path, sub_channel))
    {
        values.push_back(record->data()->left_pedal());
    }
    return values;
}

const std::string DATA_COLLECTION = "pedals";
const std::string METADATA_COLLECTION = core::tracker_metadata_collection_id(DATA_COLLECTION);

//! A producer and the live impl reading it, with the fake runtime between them.
struct LiveSession
{
    FakeTensorRuntime runtime;
    std::shared_ptr<core::MetadataTestTracker> tracker = std::make_shared<core::MetadataTestTracker>(DATA_COLLECTION);
    std::unique_ptr<mcap::McapWriter> writer;
    std::unique_ptr<core::LiveMetadataTestTrackerImpl> impl;
    int64_t now_ns = 0;

    //! Records to `recording_path` unless it is empty.
    explicit LiveSession(const std::string& recording_path = {})
    {
        std::unique_ptr<core::MetadataTestTrackerMcapChannels> channels;
        if (!recording_path.empty())
        {
            writer = open_writer(recording_path);
            channels = core::LiveMetadataTestTrackerImpl::create_mcap_channels(*writer, BASE_NAME);
        }
        impl = std::make_unique<core::LiveMetadataTestTrackerImpl>(runtime.handles(), tracker.get(), std::move(channels));
    }

    void start_data()
    {
        runtime.create(DATA_COLLECTION, static_cast<uint32_t>(tracker->max_flatbuffer_size()));
    }

    void start_metadata(uint32_t sample_size = core::TRACKER_METADATA_MAX_SIZE)
    {
        runtime.create(METADATA_COLLECTION, sample_size);
    }

    void push_data(float left_pedal)
    {
        core::Generic3AxisPedalOutputT native;
        native.left_pedal = left_pedal;
        runtime.push(DATA_COLLECTION, core::pack<core::Generic3AxisPedalOutput>(native).buffer(), ++now_ns);
    }

    void push_metadata(const core::Serialized<core::TestTrackerMetadata>& metadata)
    {
        runtime.push(METADATA_COLLECTION, metadata.buffer(), ++now_ns);
    }

    //! Runs one tick and returns what it published.
    std::optional<float> step()
    {
        impl->update(++now_ns);
        return published(impl->get_data());
    }

    void finish_recording()
    {
        impl.reset();
        writer->close();
    }
};

#endif

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
        if (view.channel->topic != core::mcap_topic(BASE_NAME, core::MetadataTestRecordingTraits::metadata_channels[0]))
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

#if defined(XR_USE_TIMESPEC)

TEST_CASE("Live publishes and records data only once the producer's metadata arrives", "[unit][tracker_metadata]")
{
    const auto path = temp_mcap_path("test_tracker_metadata");
    const TempFileCleanup cleanup(path);
    {
        LiveSession live(path);
        live.start_data();
        live.push_data(0.1f);
        CHECK_FALSE(live.step());
        CHECK_FALSE(live.impl->get_metadata());

        live.start_metadata();
        live.push_metadata(make_metadata("glove"));
        live.push_data(0.2f);
        CHECK(live.step() == 0.2f);
        REQUIRE(live.impl->get_metadata());
        CHECK(live.impl->get_metadata()->label()->str() == "glove");

        live.push_metadata(make_metadata("glove"));
        live.push_data(0.3f);
        CHECK(live.step() == 0.3f);
        live.finish_recording();
    }

    using Traits = core::MetadataTestRecordingTraits;
    const auto metadata = read_channel<core::TestTrackerMetadata>(path, Traits::metadata_channels[0]);
    REQUIRE(metadata.size() == 1);
    CHECK(metadata[0]->label()->str() == "glove");
    CHECK(recorded_left_pedals(path, Traits::recording_channels[0]) == std::vector<float>{ 0.2f, 0.3f });
    CHECK(recorded_left_pedals(path, Traits::recording_channels[1]) == std::vector<float>{ 0.2f, 0.3f });

    auto replay = open_replay(path);
    REQUIRE(replay->get_metadata());
    CHECK(replay->get_metadata()->label()->str() == "glove");
    replay->update(0);
    CHECK(published(replay->get_data()) == 0.2f);
    replay->update(0);
    CHECK(published(replay->get_data()) == 0.3f);
}

TEST_CASE("Metadata that arrives before the data collection lets its first sample through", "[unit][tracker_metadata]")
{
    LiveSession live;
    live.start_metadata();
    live.push_metadata(make_metadata("glove"));
    CHECK_FALSE(live.step());

    live.start_data();
    live.push_data(0.5f);
    CHECK(live.step() == 0.5f);
}

TEST_CASE("A restarted producer's data waits for its metadata, however it is padded", "[unit][tracker_metadata]")
{
    const auto metadata = make_metadata("glove");
    LiveSession live;
    live.start_metadata();
    live.start_data();
    live.push_metadata(metadata);
    live.push_data(0.1f);
    CHECK(live.step() == 0.1f);

    live.runtime.destroy(METADATA_COLLECTION);
    live.runtime.destroy(DATA_COLLECTION);
    CHECK_FALSE(live.step());

    live.start_data();
    live.push_data(0.2f);
    CHECK_FALSE(live.step());

    live.start_metadata(static_cast<uint32_t>(metadata.buffer().size()));
    live.push_metadata(metadata);
    live.push_data(0.3f);
    CHECK(live.step() == 0.3f);
}

TEST_CASE("Different metadata withholds the data for the rest of the session", "[unit][tracker_metadata]")
{
    LiveSession live;
    live.start_metadata();
    live.start_data();
    live.push_metadata(make_metadata("glove"));
    live.push_data(0.1f);
    CHECK(live.step() == 0.1f);

    SECTION("in a tick of its own")
    {
        live.push_metadata(make_metadata("other"));
        live.push_data(0.2f);
        CHECK_FALSE(live.step());
    }
    SECTION("followed by the adopted one in the same tick")
    {
        live.push_metadata(make_metadata("other"));
        live.push_metadata(make_metadata("glove"));
        live.push_data(0.2f);
        CHECK_FALSE(live.step());
    }

    live.push_metadata(make_metadata("glove"));
    live.push_data(0.3f);
    CHECK_FALSE(live.step());
    CHECK(live.impl->get_metadata()->label()->str() == "glove");
}

TEST_CASE("A metadata sample that is not a valid table is an error naming its collection", "[unit][tracker_metadata]")
{
    LiveSession live;
    live.start_metadata();
    const std::vector<uint8_t> garbage{ 0xff, 0xff, 0xff, 0x7f };
    live.runtime.push(METADATA_COLLECTION, garbage, 1);
    CHECK_THROWS_WITH(live.step(), Catch::Matchers::ContainsSubstring(METADATA_COLLECTION));
}

#endif
