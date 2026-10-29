// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the sample stream: TrackerSampleStreamWriter splits what an McapWriter emits
// into setup and per-tick samples, merge_samples_records folds two ticks' samples into one, TrackerSampleStreamReader
// keeps the newest record per topic, and McapTrackerViewers reads that through a stream source.

#include "mcap_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mcap/sample_stream.hpp>
#include <mcap/writer.hpp>
#include <schema/hand_generated.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace mcap_test;

constexpr int64_t kMaxAge = 250'000'000;

//! An McapWriter recording into a stream, opened the way DeviceIOSession opens one.
struct StreamRecorder
{
    // Declared first so it outlives the writer, which writes the end of the recording on close.
    core::TrackerSampleStreamWriter out;
    mcap::McapWriter writer;

    StreamRecorder()
    {
        writer.open(out, core::TrackerSampleStreamWriter::writer_options());
    }
};

core::Serialized<core::HeadPoseRecord> head_record(float x)
{
    core::HeadPoseT data;
    data.is_valid = true;
    data.pose = std::make_shared<core::Pose>(core::Point(x, 0.0f, 0.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    return core::pack_record<core::HeadPoseRecord>(&data, core::DeviceDataTimestamp(1, 1, 1));
}

float head_x(const std::vector<uint8_t>& bytes)
{
    return flatbuffers::GetRoot<core::HeadPoseRecord>(bytes.data())->data()->pose()->position().x();
}

} // namespace

TEST_CASE("TrackerSampleStreamWriter: setup is taken once, samples carry only messages", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels head(rec.writer, "0", { "head" });

    head.write(0, head_record(1.0f));
    const auto setup = rec.out.take_setup();
    const auto first = rec.out.take_samples();
    CHECK_FALSE(setup.empty());
    CHECK_FALSE(first.empty());

    head.write(0, head_record(2.0f));
    CHECK(rec.out.take_setup().empty());
    const auto second = rec.out.take_samples();
    CHECK_FALSE(second.empty());
    CHECK(rec.out.take_samples().empty());

    core::TrackerSampleStreamReader reader;
    reader.add_setup(setup);
    REQUIRE(reader.add_samples(first, 1, 100));
    REQUIRE(reader.add_samples(second, 2, 200));
    REQUIRE(reader.snapshot(200, kMaxAge));

    const auto latest = reader.latest("0/head");
    REQUIRE(latest.bytes);
    CHECK(head_x(*latest.bytes) == 2.0f);
    REQUIRE(reader.channel("0/head") != nullptr);
    CHECK(reader.channel("0/head")->message_encoding == "flatbuffer");
}

TEST_CASE("TrackerSampleStreamWriter: the topic filter drops other channels", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    rec.out.set_topic_filter({ "0/head" });
    HeadChannels kept(rec.writer, "0", { "head" });
    HeadChannels dropped(rec.writer, "1", { "head" });

    kept.write(0, head_record(1.0f));
    dropped.write(0, head_record(9.0f));

    core::TrackerSampleStreamReader reader;
    reader.add_setup(rec.out.take_setup());
    REQUIRE(reader.add_samples(rec.out.take_samples(), 1, 100));
    REQUIRE(reader.snapshot(100, kMaxAge));

    CHECK(reader.latest("0/head").bytes);
    CHECK_FALSE(reader.latest("1/head").bytes);
    CHECK(reader.channel("1/head") == nullptr);
    CHECK(reader.dropped_messages() == 0);
}

TEST_CASE("TrackerSampleStreamWriter: add_record adds a message on a kept topic, after the session's",
          "[mcap][sample_stream]")
{
    StreamRecorder rec;
    rec.out.set_topic_filter({ "0/head", "2/head" });
    HeadChannels kept(rec.writer, "0", { "head" });
    HeadChannels dropped(rec.writer, "1", { "head" });
    const auto no_data = core::pack_record<core::HeadPoseRecord>(nullptr, core::DeviceDataTimestamp(2, 2, 2));

    // A topic has a channel once its first record declared it.
    CHECK_FALSE(rec.out.add_record("0/head", no_data.buffer(), 2));
    kept.write(0, head_record(1.0f));
    dropped.write(0, head_record(9.0f));
    const auto setup = rec.out.take_setup();
    const auto older = rec.out.take_samples();

    kept.write(0, head_record(2.0f));
    CHECK(rec.out.add_record("0/head", no_data.buffer(), 2));
    CHECK_FALSE(rec.out.add_record("1/head", no_data.buffer(), 2));
    CHECK_FALSE(rec.out.add_record("2/head", no_data.buffer(), 2));
    const auto newer = rec.out.take_samples();

    // The added record is the topic's newest, also once merged into an unsent tick.
    for (const auto& samples : { newer, core::merge_samples_records(older, newer) })
    {
        core::TrackerSampleStreamReader reader;
        reader.add_setup(setup);
        REQUIRE(reader.add_samples(samples, 1, 100));
        REQUIRE(reader.snapshot(100, kMaxAge));
        const auto latest = reader.latest("0/head");
        REQUIRE(latest.bytes);
        CHECK(flatbuffers::GetRoot<core::HeadPoseRecord>(latest.bytes->data())->data() == nullptr);
        CHECK(reader.dropped_messages() == 0);
    }
}

TEST_CASE("TrackerSampleStreamReader: a topic keeps its newest record across ticks", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels a(rec.writer, "0", { "head" });
    HeadChannels b(rec.writer, "1", { "head" });

    core::TrackerSampleStreamReader reader;

    a.write(0, head_record(1.0f));
    b.write(0, head_record(10.0f));
    reader.add_setup(rec.out.take_setup());
    REQUIRE(reader.add_samples(rec.out.take_samples(), 1, 100));
    REQUIRE(reader.snapshot(100, kMaxAge));
    const uint64_t b_sequence = reader.latest("1/head").sequence;

    a.write(0, head_record(2.0f));
    REQUIRE(reader.add_samples(rec.out.take_samples(), 2, 200));
    REQUIRE(reader.snapshot(200, kMaxAge));

    CHECK(head_x(*reader.latest("0/head").bytes) == 2.0f);
    CHECK(head_x(*reader.latest("1/head").bytes) == 10.0f);
    CHECK(reader.latest("1/head").sequence == b_sequence);
}

TEST_CASE("TrackerSampleStreamReader: a stale stream publishes no records", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels head(rec.writer, "0", { "head" });
    core::TrackerSampleStreamReader reader;

    head.write(0, head_record(1.0f));
    reader.add_setup(rec.out.take_setup());
    REQUIRE(reader.add_samples(rec.out.take_samples(), 1, 1'000));

    CHECK_FALSE(reader.snapshot(1'000 + kMaxAge + 1, kMaxAge));
    CHECK_FALSE(reader.latest("0/head").bytes);
    CHECK_FALSE(reader.published_tick_time_ns().has_value());
    CHECK(reader.channel("0/head") != nullptr);

    head.write(0, head_record(2.0f));
    REQUIRE(reader.add_samples(rec.out.take_samples(), 2, 2'000 + kMaxAge));
    CHECK(reader.snapshot(2'000 + kMaxAge, kMaxAge));
    CHECK(head_x(*reader.latest("0/head").bytes) == 2.0f);
    CHECK(reader.published_tick_time_ns() == 2'000 + kMaxAge);
}

TEST_CASE("TrackerSampleStreamReader: a stream with no samples yet is stale", "[mcap][sample_stream]")
{
    core::TrackerSampleStreamReader reader;
    CHECK_FALSE(reader.snapshot(1'000, kMaxAge));
    CHECK_FALSE(reader.newest_tick_time_ns().has_value());
    CHECK_FALSE(reader.published_tick_time_ns().has_value());
}

TEST_CASE("TrackerSampleStreamReader: orders ticks by tick_seq, not by tick time", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels head(rec.writer, "0", { "head" });
    core::TrackerSampleStreamReader reader;

    head.write(0, head_record(1.0f));
    reader.add_setup(rec.out.take_setup());
    const auto first = rec.out.take_samples();
    head.write(0, head_record(2.0f));
    const auto second = rec.out.take_samples();
    head.write(0, head_record(3.0f));
    const auto third = rec.out.take_samples();

    REQUIRE(reader.add_samples(second, 2, 200));
    CHECK_FALSE(reader.add_samples(first, 1, 300));
    CHECK_FALSE(reader.add_samples(first, 2, 300));
    REQUIRE(reader.snapshot(200, kMaxAge));
    CHECK(head_x(*reader.latest("0/head").bytes) == 2.0f);

    // A newer tick whose mapped tick time is older, as after the clock offset estimate moves.
    REQUIRE(reader.add_samples(third, 3, 150));
    CHECK(reader.newest_tick_time_ns() == 150);
    REQUIRE(reader.snapshot(200, kMaxAge));
    CHECK(head_x(*reader.latest("0/head").bytes) == 3.0f);
    CHECK(reader.published_tick_time_ns() == 150);
    CHECK_FALSE(reader.snapshot(150 + kMaxAge + 1, kMaxAge));
}

TEST_CASE("TrackerSampleStreamReader: the first tick is taken whatever its tick_seq", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels head(rec.writer, "0", { "head" });
    core::TrackerSampleStreamReader reader;

    head.write(0, head_record(1.0f));
    reader.add_setup(rec.out.take_setup());
    REQUIRE(reader.add_samples(rec.out.take_samples(), 0, 100));
    CHECK_FALSE(reader.add_samples({}, 0, 200));
    CHECK(reader.add_samples({}, 1, 200));
}

TEST_CASE("TrackerSampleStreamReader: subscribed topics are sorted and unique", "[mcap][sample_stream]")
{
    core::TrackerSampleStreamReader reader;
    CHECK(reader.subscribed_topics().empty());

    reader.subscribe("1/head");
    reader.subscribe("0/right_hand");
    reader.subscribe("1/head");
    reader.subscribe("0/left_hand");
    CHECK(reader.subscribed_topics() == std::vector<std::string>{ "0/left_hand", "0/right_hand", "1/head" });
}

TEST_CASE("TrackerSampleStreamReader: drops a message on an undeclared channel", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels head(rec.writer, "0", { "head" });
    head.write(0, head_record(1.0f));

    core::TrackerSampleStreamReader reader;
    REQUIRE(reader.add_samples(rec.out.take_samples(), 1, 100));
    REQUIRE(reader.snapshot(100, kMaxAge));
    CHECK(reader.dropped_messages() == 1);
    CHECK_FALSE(reader.latest("0/head").bytes);
}

TEST_CASE("TrackerSampleStreamReader: refuses bytes that end inside a record", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels head(rec.writer, "0", { "head" });
    head.write(0, head_record(1.0f));

    core::TrackerSampleStreamReader reader;
    auto setup = rec.out.take_setup();
    auto samples = rec.out.take_samples();
    reader.add_setup(setup);

    samples.pop_back();
    CHECK_THROWS_AS(reader.add_samples(samples, 1, 100), std::runtime_error);
    setup.pop_back();
    CHECK_THROWS_AS(reader.add_setup(setup), std::runtime_error);
}

TEST_CASE("McapTrackerViewers: a stream source reads the newest record until a newer one arrives",
          "[mcap][sample_stream][tracker_viewers]")
{
    StreamRecorder rec;
    HeadChannels head(rec.writer, "0", { "head" });
    core::TrackerSampleStreamReader reader;
    HeadViewers viewers(core::TrackerSampleSource(reader), "0", { "head" });

    CHECK_FALSE(viewers.read(0));

    head.write(0, head_record(1.0f));
    reader.add_setup(rec.out.take_setup());
    REQUIRE(reader.add_samples(rec.out.take_samples(), 1, 100));
    REQUIRE(reader.snapshot(100, kMaxAge));

    const auto first = viewers.read(0);
    REQUIRE(first);
    CHECK(first->data()->pose()->position().x() == 1.0f);
    CHECK(viewers.read(0).get() == first.get());

    head.write(0, head_record(2.0f));
    REQUIRE(reader.add_samples(rec.out.take_samples(), 2, 200));
    REQUIRE(reader.snapshot(200, kMaxAge));
    const auto second = viewers.read(0);
    REQUIRE(second);
    CHECK(second->data()->pose()->position().x() == 2.0f);

    CHECK_FALSE(reader.snapshot(200 + kMaxAge + 1, kMaxAge));
    CHECK_FALSE(viewers.read(0));
}

TEST_CASE("McapTrackerViewers: a stream source subscribes to the topics it reads",
          "[mcap][sample_stream][tracker_viewers]")
{
    core::TrackerSampleStreamReader reader;
    HeadViewers first(core::TrackerSampleSource(reader), "1", { "head" });
    HeadViewers second(core::TrackerSampleSource(reader), "0", { "head", "other" });

    CHECK(reader.subscribed_topics() == std::vector<std::string>{ "0/head", "0/other", "1/head" });
}

TEST_CASE("McapTrackerViewers: a stream source checks a declared schema before any record arrives",
          "[mcap][sample_stream][tracker_viewers]")
{
    StreamRecorder rec;
    // "0/head" declared with the hand record's schema, which a head viewer cannot read.
    core::McapTrackerChannels<core::HandPoseRecord> wrong(rec.writer, "0", { "head" });
    HeadChannels right(rec.writer, "1", { "head" });

    core::TrackerSampleStreamReader reader;
    HeadViewers wrong_viewers(core::TrackerSampleSource(reader), "0", { "head" });
    HeadViewers right_viewers(core::TrackerSampleSource(reader), "1", { "head" });

    // Nothing declared yet: nothing to check.
    CHECK_FALSE(wrong_viewers.read(0));

    // The writer declares a channel with its first message; only the declarations arrive.
    wrong.write(0, core::pack_record<core::HandPoseRecord>(nullptr, core::DeviceDataTimestamp(1, 1, 1)));
    right.write(0, head_record(1.0f));
    reader.add_setup(rec.out.take_setup());
    CHECK_FALSE(reader.snapshot(100, kMaxAge));
    CHECK_THROWS_WITH(wrong_viewers.read(0), Catch::Matchers::ContainsSubstring("MCAP schema mismatch on '0/head'"));
    CHECK_FALSE(right_viewers.read(0));
}

TEST_CASE("merge_samples_records: keeps the last record per channel", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels a(rec.writer, "0", { "head" });
    HeadChannels b(rec.writer, "1", { "head" });

    a.write(0, head_record(1.0f));
    b.write(0, head_record(10.0f));
    const auto setup = rec.out.take_setup();
    const auto older = rec.out.take_samples();

    a.write(0, head_record(2.0f));
    a.write(0, head_record(3.0f));
    const auto newer = rec.out.take_samples();

    const auto merged = core::merge_samples_records(older, newer);
    // One record per channel, each the size of the one it replaced.
    CHECK(merged.size() == older.size());

    core::TrackerSampleStreamReader reader;
    reader.add_setup(setup);
    REQUIRE(reader.add_samples(merged, 1, 100));
    REQUIRE(reader.snapshot(100, kMaxAge));
    CHECK(head_x(*reader.latest("0/head").bytes) == 3.0f);
    CHECK(head_x(*reader.latest("1/head").bytes) == 10.0f);

    CHECK(core::merge_samples_records(older, {}) == older);
    CHECK(core::merge_samples_records({}, older) == older);
    CHECK(core::merge_samples_records(newer, {}).size() == newer.size() / 2);
    CHECK(core::merge_samples_records({}, {}).empty());
}

TEST_CASE("merge_samples_records: refuses incomplete and non-Message records", "[mcap][sample_stream]")
{
    StreamRecorder rec;
    HeadChannels head(rec.writer, "0", { "head" });
    head.write(0, head_record(1.0f));
    const auto setup = rec.out.take_setup();
    const auto samples = rec.out.take_samples();

    auto truncated = samples;
    truncated.pop_back();
    CHECK_THROWS_AS(core::merge_samples_records(truncated, samples), std::runtime_error);
    CHECK_THROWS_AS(core::merge_samples_records(samples, truncated), std::runtime_error);
    CHECK_THROWS_WITH(core::merge_samples_records(samples, setup),
                      Catch::Matchers::ContainsSubstring("a record other than a Message"));
}
