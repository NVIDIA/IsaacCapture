// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for RobotStateStore, the server end of robot state. The payloads are the ones of the robot-state link
// messages: written by an McapWriter on a TrackerSampleStreamWriter, or laid out by hand where a case needs a
// payload no writer makes.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mcap/sample_stream.hpp>
#include <mcap/writer.hpp>
#include <robot_state_store/robot_state_store.hpp>

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using Bytes = std::vector<uint8_t>;
using Metadata = std::vector<std::pair<std::string, std::string>>;
using Catch::Matchers::ContainsSubstring;

constexpr int64_t kMs = 1'000'000;
constexpr int64_t kSecond = 1'000 * kMs;
constexpr uint32_t kClockValid = 1u << 0;
constexpr uint32_t kClockTwoWay = 1u << 1;

template <typename T>
void put(Bytes& out, T value)
{
    const auto bits = static_cast<std::make_unsigned_t<T>>(value);
    for (size_t i = 0; i < sizeof(T); ++i)
    {
        out.push_back(static_cast<uint8_t>(bits >> (8 * i)));
    }
}

void put_text(Bytes& out, std::string_view text)
{
    put<uint32_t>(out, static_cast<uint32_t>(text.size()));
    out.insert(out.end(), text.begin(), text.end());
}

Bytes concat(std::initializer_list<Bytes> parts)
{
    Bytes out;
    for (const Bytes& part : parts)
    {
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

//! An MCAP record: opcode, u64 body length, body.
Bytes record(uint8_t opcode, const Bytes& body)
{
    Bytes out{ opcode };
    put<uint64_t>(out, body.size());
    return concat({ out, body });
}

Bytes schema_record(uint16_t id, std::string_view name, std::string_view encoding, std::string_view data)
{
    Bytes body;
    put<uint16_t>(body, id);
    put_text(body, name);
    put_text(body, encoding);
    put_text(body, data);
    return record(0x03, body);
}

Bytes channel_record(
    uint16_t id, uint16_t schema_id, std::string_view topic, std::string_view encoding, const Metadata& metadata)
{
    Bytes pairs;
    for (const auto& [key, value] : metadata)
    {
        put_text(pairs, key);
        put_text(pairs, value);
    }
    Bytes body;
    put<uint16_t>(body, id);
    put<uint16_t>(body, schema_id);
    put_text(body, topic);
    put_text(body, encoding);
    put<uint32_t>(body, static_cast<uint32_t>(pairs.size()));
    return record(0x04, concat({ body, pairs }));
}

Bytes message_record(uint16_t channel, int64_t sample_time_ns, std::string_view data, uint32_t sequence = 1)
{
    Bytes body;
    put<uint16_t>(body, channel);
    put<uint32_t>(body, sequence);
    put<int64_t>(body, sample_time_ns);
    put<int64_t>(body, sample_time_ns + 1);
    body.insert(body.end(), data.begin(), data.end());
    return record(0x05, body);
}

//! The prefix of a ROBOT_STATE_SAMPLES or ROBOT_STATE_RECORDS payload.
struct Prefix
{
    uint64_t publish_seq = 1;
    int64_t robot_minus_server_ns = 0;
    uint32_t flags = kClockValid | kClockTwoWay;
};

Bytes samples_payload(const Prefix& prefix, const Bytes& records)
{
    Bytes out;
    put<uint64_t>(out, prefix.publish_seq);
    put<int64_t>(out, 0);
    put<int64_t>(out, prefix.robot_minus_server_ns);
    put<uint32_t>(out, prefix.flags);
    put<uint32_t>(out, 0);
    return concat({ out, records });
}

constexpr uint16_t kDescription = 1;
constexpr uint16_t kArm = 2;
constexpr uint16_t kStatus = 3;

//! The setup of a robot with a description, an arm that goes stale after 100 ms and a status after 2 s.
Bytes robot_setup()
{
    return concat({
        schema_record(kDescription, "core.RobotDescriptionRecord", "flatbuffer", "description-schema"),
        channel_record(
            kDescription, kDescription, "robot_description", "flatbuffer",
            { { "delivery", "in_order" }, { "rate_hz", "0" }, { "shape", "description" }, { "stale_after_ns", "0" } }),
        schema_record(kArm, "core.JointStateOutputRecord", "flatbuffer", "joints-schema"),
        channel_record(
            kArm, kArm, "arm", "flatbuffer",
            { { "delivery", "latest" }, { "rate_hz", "20" }, { "shape", "joints" }, { "stale_after_ns", "100000000" } }),
        schema_record(kStatus, "core.RobotStatusRecord", "flatbuffer", "status-schema"),
        channel_record(kStatus, kStatus, "status", "flatbuffer",
                       { { "delivery", "latest" }, { "shape", "status" }, { "stale_after_ns", "2000000000" } }),
    });
}

std::string text(const core::RobotRecord& record)
{
    return record.data ? std::string(record.data->begin(), record.data->end()) : std::string("<no record>");
}

std::string text(const Bytes& bytes)
{
    return std::string(bytes.begin(), bytes.end());
}

std::vector<std::string> names(const core::RobotStateStore& store)
{
    std::vector<std::string> out;
    for (const core::RobotSeriesInfo& info : store.series())
    {
        out.push_back(info.name);
    }
    return out;
}

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

    uint16_t add_series(std::string_view topic,
                        std::string_view schema_name,
                        std::string_view schema_data,
                        mcap::KeyValueMap metadata)
    {
        mcap::Schema schema(schema_name, "flatbuffer", schema_data);
        writer.addSchema(schema);
        mcap::Channel channel(topic, "flatbuffer", schema.id, metadata);
        writer.addChannel(channel);
        return channel.id;
    }

    void write(uint16_t channel, int64_t sample_time_ns, std::string_view data)
    {
        mcap::Message message;
        message.channelId = channel;
        message.sequence = ++sequence;
        message.logTime = static_cast<uint64_t>(sample_time_ns);
        message.publishTime = static_cast<uint64_t>(sample_time_ns + 1);
        message.data = reinterpret_cast<const std::byte*>(data.data());
        message.dataSize = data.size();
        REQUIRE(writer.write(message).ok());
    }

    uint32_t sequence = 0;
};

} // namespace

TEST_CASE("RobotStateStore: a new store is empty and not connected", "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    CHECK_FALSE(store.connected());
    CHECK(store.session_count() == 0);
    CHECK(store.generation() == 0);
    CHECK(store.setup_generation() == 1);
    CHECK(store.dropped_records() == 0);
    CHECK(store.series().empty());
    CHECK_FALSE(store.series_info("arm"));

    const core::RobotRecord none = store.latest("arm", 5 * kSecond);
    CHECK_FALSE(none.data);
    CHECK_FALSE(none.fresh);
    CHECK_FALSE(none.age_ns);
    CHECK(none.sequence == 0);
    CHECK(core::kRobotDescriptionTopic == "robot_description");
}

TEST_CASE("RobotStateStore: setup and messages written by an McapWriter are read", "[robot_state_store][unit]")
{
    StreamRecorder rec;
    const uint16_t description =
        rec.add_series("robot_description", "core.RobotDescriptionRecord", "description-schema",
                       { { "shape", "description" }, { "delivery", "in_order" }, { "stale_after_ns", "0" } });
    const uint16_t arm = rec.add_series(
        "arm", "core.JointStateOutputRecord", "joints-schema",
        { { "shape", "joints" }, { "delivery", "latest" }, { "stale_after_ns", "500000000" }, { "rate_hz", "20.5" } });

    // The writer declares a channel with its first message.
    rec.write(description, 10 * kSecond, "what the robot is");
    const Bytes first_setup = rec.out.take_setup();
    const Bytes first_samples = rec.out.take_samples();
    rec.write(arm, 11 * kSecond, "angles 1");
    rec.write(arm, 12 * kSecond, "angles 2");
    const Bytes second_setup = rec.out.take_setup();
    const Bytes second_samples = rec.out.take_samples();
    REQUIRE_FALSE(first_setup.empty());
    REQUIRE_FALSE(second_setup.empty());

    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(first_setup);
    store.add_samples(samples_payload({ .publish_seq = 1 }, first_samples), 100 * kSecond);
    store.add_setup(second_setup);
    store.add_samples(samples_payload({ .publish_seq = 2 }, second_samples), 101 * kSecond);

    CHECK(names(store) == std::vector<std::string>{ "robot_description", "arm" });
    CHECK(store.dropped_records() == 0);

    const auto info = store.series_info("arm");
    REQUIRE(info);
    CHECK(info->name == "arm");
    CHECK(info->shape == "joints");
    CHECK(info->schema_name == "core.JointStateOutputRecord");
    CHECK(info->schema_encoding == "flatbuffer");
    CHECK(text(info->schema_data) == "joints-schema");
    CHECK(info->message_encoding == "flatbuffer");
    CHECK(info->stale_after_ns == 500 * kMs);
    CHECK_FALSE(info->in_order);
    CHECK(info->rate_hz == 20.5);

    const auto description_info = store.series_info("robot_description");
    REQUIRE(description_info);
    CHECK(description_info->shape == "description");
    CHECK(description_info->in_order);
    CHECK(description_info->stale_after_ns == 0);
    CHECK(text(description_info->schema_data) == "description-schema");

    const core::RobotRecord newest = store.latest("arm", 12 * kSecond);
    CHECK(text(newest) == "angles 2");
    CHECK(newest.sequence == 2);
    CHECK(newest.sample_time_ns == 12 * kSecond);
    CHECK(newest.publish_time_ns == 12 * kSecond + 1);
    CHECK(newest.receive_time_ns == 101 * kSecond);
    CHECK(text(store.latest("robot_description", 12 * kSecond)) == "what the robot is");
}

TEST_CASE("RobotStateStore: channel metadata takes defaults for missing, unknown and unreadable values",
          "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(concat({
        schema_record(1, "s", "flatbuffer", ""),
        channel_record(1, 1, "bare", "flatbuffer", {}),
        channel_record(
            2, 1, "odd", "flatbuffer",
            { { "stale_after_ns", "soon" }, { "rate_hz", "fast" }, { "delivery", "by_post" }, { "colour", "blue" } }),
        channel_record(3, 1, "negative", "flatbuffer", { { "stale_after_ns", "-5" }, { "rate_hz", "-1" } }),
        channel_record(4, 1, "exact", "flatbuffer",
                       { { "stale_after_ns", "9223372036854775807" }, { "rate_hz", "1e3" }, { "shape", "force" } }),
    }));

    for (const char* name : { "bare", "odd", "negative" })
    {
        INFO(name);
        const auto info = store.series_info(name);
        REQUIRE(info);
        CHECK(info->shape.empty());
        CHECK(info->stale_after_ns == 0);
        CHECK_FALSE(info->in_order);
        CHECK(info->rate_hz == 0.0);
    }
    const auto exact = store.series_info("exact");
    REQUIRE(exact);
    CHECK(exact->shape == "force");
    CHECK(exact->stale_after_ns == INT64_MAX);
    CHECK(exact->rate_hz == 1000.0);
}

TEST_CASE("RobotStateStore: the age of a record is measured through the clock estimate that came with it",
          "[robot_state_store][unit]")
{
    // The robot's clock reads 1000 s more than this machine's.
    constexpr int64_t kRobotAhead = 1'000 * kSecond;
    constexpr int64_t kSampleHere = 50 * kSecond;
    constexpr int64_t kSampleRobot = kSampleHere + kRobotAhead;

    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(robot_setup());

    SECTION("a two-way estimate")
    {
        store.add_samples(samples_payload({ .publish_seq = 1, .robot_minus_server_ns = kRobotAhead },
                                          message_record(kArm, kSampleRobot, "a")),
                          kSampleHere + 3 * kMs);
        const core::RobotRecord record = store.latest("arm", kSampleHere + 30 * kMs);
        REQUIRE(record.age_ns);
        CHECK(*record.age_ns == 30 * kMs);
        CHECK(record.receive_age_ns == 27 * kMs);
        CHECK(record.clock_two_way);
        CHECK(record.fresh);
        CHECK(record.sample_time_ns == kSampleRobot);
    }

    SECTION("a robot clock behind this machine's")
    {
        store.add_samples(samples_payload({ .publish_seq = 1, .robot_minus_server_ns = -kRobotAhead },
                                          message_record(kArm, kSampleHere - kRobotAhead, "a")),
                          kSampleHere + 3 * kMs);
        const core::RobotRecord record = store.latest("arm", kSampleHere + 30 * kMs);
        REQUIRE(record.age_ns);
        CHECK(*record.age_ns == 30 * kMs);
    }

    SECTION("a one-way estimate gives an age and says it is one-way")
    {
        // A one-way estimate is too large by the shortest delay (4 ms here), so the age reads 4 ms more.
        store.add_samples(
            samples_payload({ .publish_seq = 1, .robot_minus_server_ns = kRobotAhead + 4 * kMs, .flags = kClockValid },
                            message_record(kArm, kSampleRobot, "a")),
            kSampleHere + 4 * kMs);
        const core::RobotRecord record = store.latest("arm", kSampleHere + 30 * kMs);
        REQUIRE(record.age_ns);
        CHECK(*record.age_ns == 34 * kMs);
        CHECK_FALSE(record.clock_two_way);
    }

    SECTION("no estimate: no age, and freshness follows the receive age")
    {
        store.add_samples(samples_payload({ .publish_seq = 1, .robot_minus_server_ns = 12345, .flags = kClockTwoWay },
                                          message_record(kArm, kSampleRobot, "a")),
                          kSampleHere + 3 * kMs);
        const core::RobotRecord early = store.latest("arm", kSampleHere + 103 * kMs);
        CHECK_FALSE(early.age_ns);
        CHECK_FALSE(early.clock_two_way);
        CHECK(early.receive_age_ns == 100 * kMs);
        CHECK(early.fresh);
        CHECK_FALSE(store.latest("arm", kSampleHere + 104 * kMs).fresh);
    }

    SECTION("an age is never negative")
    {
        store.add_samples(samples_payload({ .publish_seq = 1, .robot_minus_server_ns = kRobotAhead },
                                          message_record(kArm, kSampleRobot, "a")),
                          kSampleHere + 3 * kMs);
        const core::RobotRecord record = store.latest("arm", kSampleHere - 5 * kMs);
        REQUIRE(record.age_ns);
        CHECK(*record.age_ns == 0);
        CHECK(record.receive_age_ns == 0);
        CHECK(record.fresh);
    }

    SECTION("a time stamp ahead of the robot's clock: the record is as old as the time since it arrived")
    {
        // A robot app that stamps a reading with wall-clock time instead of its monotonic clock.
        constexpr int64_t kWallClock = 1'700'000'000 * kSecond;
        store.add_samples(samples_payload({ .publish_seq = 1, .robot_minus_server_ns = kRobotAhead },
                                          message_record(kArm, kWallClock, "a")),
                          kSampleHere + 3 * kMs);
        const core::RobotRecord early = store.latest("arm", kSampleHere + 30 * kMs);
        REQUIRE(early.age_ns);
        CHECK(*early.age_ns == 27 * kMs);
        CHECK(early.fresh);

        const core::RobotRecord late = store.latest("arm", kSampleHere + 5 * kSecond);
        REQUIRE(late.age_ns);
        CHECK(*late.age_ns == 5 * kSecond - 3 * kMs);
        CHECK(late.receive_age_ns == 5 * kSecond - 3 * kMs);
        CHECK_FALSE(late.fresh);
    }

    SECTION("times at the ends of the range do not overflow")
    {
        store.add_samples(samples_payload({ .publish_seq = 1, .robot_minus_server_ns = INT64_MAX },
                                          message_record(kArm, INT64_MIN, "a")),
                          INT64_MIN);
        const core::RobotRecord record = store.latest("arm", INT64_MAX);
        REQUIRE(record.age_ns);
        CHECK(*record.age_ns == INT64_MAX);
        CHECK(record.receive_age_ns == INT64_MAX);
        CHECK_FALSE(record.fresh);
    }
}

TEST_CASE("RobotStateStore: each series goes stale by its own limit and the description never does",
          "[robot_state_store][unit]")
{
    constexpr int64_t kSample = 10 * kSecond;

    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(robot_setup());
    store.add_samples(samples_payload({ .publish_seq = 1 }, message_record(kDescription, kSample, "what")), kSample);
    store.add_samples(samples_payload({ .publish_seq = 1 }, concat({ message_record(kArm, kSample, "angles"),
                                                                     message_record(kStatus, kSample, "ok") })),
                      kSample);

    // At the limit a record is still fresh.
    CHECK(store.latest("arm", kSample + 100 * kMs).fresh);
    CHECK_FALSE(store.latest("arm", kSample + 100 * kMs + 1).fresh);
    CHECK(store.latest("status", kSample + 100 * kMs + 1).fresh);
    CHECK(store.latest("status", kSample + 2 * kSecond).fresh);
    CHECK_FALSE(store.latest("status", kSample + 2 * kSecond + 1).fresh);

    // A stale record is still handed out, with its age.
    const core::RobotRecord stale = store.latest("arm", kSample + kSecond);
    CHECK(text(stale) == "angles");
    REQUIRE(stale.age_ns);
    CHECK(*stale.age_ns == kSecond);

    const core::RobotRecord description = store.latest("robot_description", kSample + 3'600 * kSecond);
    CHECK(text(description) == "what");
    CHECK(description.fresh);

    // A newer arm record makes only the arm fresh again.
    store.add_samples(samples_payload({ .publish_seq = 2 }, message_record(kArm, kSample + 3 * kSecond, "angles 2")),
                      kSample + 3 * kSecond);
    CHECK(store.latest("arm", kSample + 3 * kSecond + 50 * kMs).fresh);
    CHECK_FALSE(store.latest("status", kSample + 3 * kSecond + 50 * kMs).fresh);
}

TEST_CASE("RobotStateStore: a new robot session replaces the series and records of the one before",
          "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    CHECK(store.connected());
    CHECK(store.session_count() == 1);
    CHECK(store.setup_generation() == 2);
    store.add_setup(robot_setup());
    store.add_samples(samples_payload({ .publish_seq = 7 }, message_record(kDescription, 1, "six-joint arm")), 1);
    store.add_samples(samples_payload({ .publish_seq = 7 }, message_record(kArm, 1, "angles")), 1);
    const uint64_t setup_generation = store.setup_generation();
    CHECK(setup_generation > 2);

    SECTION("the robot app leaves: everything stays readable")
    {
        const uint64_t before = store.generation();
        store.end_session();
        CHECK_FALSE(store.connected());
        CHECK(store.generation() > before);
        CHECK(store.session_count() == 1);
        CHECK(store.setup_generation() == setup_generation);
        CHECK(names(store).size() == 3);
        CHECK(text(store.latest("robot_description", 3'600 * kSecond)) == "six-joint arm");
        CHECK(store.latest("robot_description", 3'600 * kSecond).fresh);
        CHECK(text(store.latest("arm", 1)) == "angles");

        // A second end_session() changes nothing.
        const uint64_t ended = store.generation();
        store.end_session();
        CHECK(store.generation() == ended);
        CHECK_FALSE(store.connected());
    }

    SECTION("another robot app connects: nothing of the first one is left")
    {
        store.end_session();
        store.begin_session();
        CHECK(store.connected());
        CHECK(store.session_count() == 2);
        CHECK(store.setup_generation() > setup_generation);
        CHECK(store.series().empty());
        CHECK_FALSE(store.latest("robot_description", 1).data);
        CHECK_FALSE(store.latest("arm", 1).data);

        // The first robot's channel ids mean nothing in the new session.
        store.add_samples(samples_payload({ .publish_seq = 1 }, message_record(kArm, 2, "stray")), 2);
        CHECK(store.dropped_records() == 1);
        CHECK(store.series().empty());

        // The other robot: its description, another series under the id the arm had, publish_seq from 1 again.
        store.add_setup(concat({
            schema_record(1, "core.RobotDescriptionRecord", "flatbuffer", "description-schema"),
            channel_record(1, 1, "robot_description", "flatbuffer", { { "shape", "description" } }),
            channel_record(kArm, 1, "left_arm", "flatbuffer", { { "shape", "joints" } }),
        }));
        store.add_samples(samples_payload({ .publish_seq = 1 }, message_record(1, 3, "two-arm robot")), 3);
        store.add_samples(samples_payload({ .publish_seq = 1 }, message_record(kArm, 3, "left angles")), 3);
        CHECK(names(store) == std::vector<std::string>{ "robot_description", "left_arm" });
        CHECK(text(store.latest("robot_description", 3)) == "two-arm robot");
        CHECK(store.latest("robot_description", 3).sequence == 1);
        CHECK(text(store.latest("left_arm", 3)) == "left angles");
        CHECK_FALSE(store.latest("arm", 3).data);
    }

    SECTION("begin_session() without an end in between also starts anew")
    {
        store.begin_session();
        CHECK(store.connected());
        CHECK(store.session_count() == 2);
        CHECK(store.series().empty());
    }
}

TEST_CASE("RobotStateStore: a record on an undeclared channel is dropped and counted", "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(robot_setup());
    store.add_samples(samples_payload({ .publish_seq = 1 },
                                      concat({ message_record(99, 5, "nobody's"), message_record(kArm, 5, "angles"),
                                               message_record(98, 5, "nobody's") })),
                      5);
    CHECK(store.dropped_records() == 2);
    CHECK(text(store.latest("arm", 5)) == "angles");
    CHECK(names(store).size() == 3);
}

TEST_CASE("RobotStateStore: a payload older than a series' record does not replace it", "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(robot_setup());
    store.add_samples(samples_payload({ .publish_seq = 5 }, message_record(kArm, 50, "fifth")), 50);

    store.add_samples(samples_payload({ .publish_seq = 4 }, message_record(kArm, 40, "fourth")), 51);
    store.add_samples(samples_payload({ .publish_seq = 5 }, message_record(kArm, 50, "fifth again")), 52);
    core::RobotRecord arm = store.latest("arm", 60);
    CHECK(text(arm) == "fifth");
    CHECK(arm.sequence == 1);
    CHECK(arm.receive_time_ns == 50);
    CHECK(store.dropped_records() == 2);

    // Other series are judged on their own: the status has no record yet.
    store.add_samples(samples_payload({ .publish_seq = 2 },
                                      concat({ message_record(kArm, 20, "second"), message_record(kStatus, 20, "ok") })),
                      53);
    CHECK(text(store.latest("arm", 60)) == "fifth");
    CHECK(text(store.latest("status", 60)) == "ok");
    CHECK(store.dropped_records() == 3);

    store.add_samples(samples_payload({ .publish_seq = 6 }, message_record(kArm, 60, "sixth")), 54);
    arm = store.latest("arm", 60);
    CHECK(text(arm) == "sixth");
    CHECK(arm.sequence == 2);
}

TEST_CASE("RobotStateStore: every record of an in-order series in one payload is taken, the last one stays",
          "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(robot_setup());
    store.add_samples(samples_payload({ .publish_seq = 1 }, concat({ message_record(kDescription, 1, "first", 1),
                                                                     message_record(kDescription, 2, "second", 2),
                                                                     message_record(kDescription, 3, "third", 3) })),
                      9);
    const core::RobotRecord description = store.latest("robot_description", 9);
    CHECK(text(description) == "third");
    CHECK(description.sequence == 3);
    CHECK(description.sample_time_ns == 3);
    CHECK(store.dropped_records() == 0);
}

TEST_CASE("RobotStateStore: a malformed payload is refused and changes nothing", "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(robot_setup());
    store.add_samples(samples_payload({ .publish_seq = 1 }, message_record(kArm, 5, "angles")), 5);
    const uint64_t generation = store.generation();
    const uint64_t setup_generation = store.setup_generation();

    const Bytes good_setup =
        concat({ schema_record(9, "s", "flatbuffer", "x"), channel_record(9, 9, "extra", "flatbuffer", {}) });
    const Bytes good_message = message_record(kArm, 6, "newer angles");

    SECTION("setup that ends inside a record")
    {
        Bytes cut = good_setup;
        cut.pop_back();
        CHECK_THROWS_AS(store.add_setup(cut), std::runtime_error);
        CHECK_THROWS_AS(store.add_setup(Bytes{ 0x04, 1, 0 }), std::runtime_error);
    }
    SECTION("setup with a field that runs past its record")
    {
        Bytes body;
        put<uint16_t>(body, 9);
        put<uint16_t>(body, 0);
        put<uint32_t>(body, 1'000);
        CHECK_THROWS_WITH(store.add_setup(record(0x04, body)), ContainsSubstring("RobotStateStore"));
    }
    SECTION("setup with a record that is neither a Schema nor a Channel")
    {
        CHECK_THROWS_AS(store.add_setup(concat({ good_setup, good_message })), std::runtime_error);
    }
    SECTION("a Channel with an undeclared Schema")
    {
        CHECK_THROWS_WITH(store.add_setup(concat({ good_setup, channel_record(10, 77, "lost", "flatbuffer", {}) })),
                          ContainsSubstring("'lost'"));
    }
    SECTION("a Channel without a topic")
    {
        CHECK_THROWS_AS(store.add_setup(channel_record(10, kArm, "", "flatbuffer", {})), std::runtime_error);
    }
    SECTION("a Schema with id 0")
    {
        CHECK_THROWS_AS(store.add_setup(schema_record(0, "s", "flatbuffer", "x")), std::runtime_error);
    }
    SECTION("samples shorter than the prefix")
    {
        CHECK_THROWS_WITH(store.add_samples(Bytes(31, 0), 6), ContainsSubstring("shorter than its prefix"));
        CHECK_THROWS_AS(store.add_samples(Bytes{}, 6), std::runtime_error);
    }
    SECTION("samples that end inside a record")
    {
        Bytes cut = concat({ good_message, good_message });
        cut.pop_back();
        CHECK_THROWS_AS(store.add_samples(samples_payload({ .publish_seq = 2 }, cut), 6), std::runtime_error);
    }
    SECTION("samples with a record other than a Message")
    {
        const Bytes mixed = concat({ good_message, good_setup });
        CHECK_THROWS_AS(store.add_samples(samples_payload({ .publish_seq = 2 }, mixed), 6), std::runtime_error);
    }
    SECTION("a Message too short for its header")
    {
        CHECK_THROWS_AS(
            store.add_samples(
                samples_payload({ .publish_seq = 2 }, concat({ good_message, record(0x05, Bytes(21, 0)) })), 6),
            std::runtime_error);
    }
    SECTION("a record above the size limit")
    {
        const std::string large((size_t{ 1 } << 20) + 1, 'x');
        CHECK_THROWS_WITH(store.add_samples(samples_payload({ .publish_seq = 2 }, message_record(kArm, 6, large)), 6),
                          ContainsSubstring("above the limit"));
    }
    SECTION("more series than the limit")
    {
        Bytes many = schema_record(9, "s", "flatbuffer", "x");
        for (uint16_t i = 0; i < 254; ++i)
        {
            const Bytes channel = channel_record(100 + i, 9, "series_" + std::to_string(i), "flatbuffer", {});
            many.insert(many.end(), channel.begin(), channel.end());
        }
        CHECK_THROWS_WITH(store.add_setup(many), ContainsSubstring("more than 256"));
    }

    CHECK(store.generation() == generation);
    CHECK(store.setup_generation() == setup_generation);
    CHECK(store.dropped_records() == 0);
    CHECK(names(store) == std::vector<std::string>{ "robot_description", "arm", "status" });
    const core::RobotRecord arm = store.latest("arm", 5);
    CHECK(text(arm) == "angles");
    CHECK(arm.sequence == 1);

    // The store goes on taking good payloads.
    store.add_setup(good_setup);
    store.add_samples(samples_payload({ .publish_seq = 2 }, good_message), 6);
    CHECK(store.series_info("extra"));
    CHECK(text(store.latest("arm", 6)) == "newer angles");
}

TEST_CASE("RobotStateStore: the series limit is reached exactly", "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    Bytes setup = schema_record(1, "s", "flatbuffer", "x");
    for (uint16_t i = 0; i < 256; ++i)
    {
        const Bytes channel = channel_record(i + 1, 1, "series_" + std::to_string(i), "flatbuffer", {});
        setup.insert(setup.end(), channel.begin(), channel.end());
    }
    store.add_setup(setup);
    CHECK(store.series().size() == 256);

    // Declaring one of them again is no new series; one more is.
    store.add_setup(channel_record(1, 1, "series_0", "flatbuffer", {}));
    CHECK_THROWS_AS(store.add_setup(channel_record(300, 1, "one_more", "flatbuffer", {})), std::runtime_error);
    CHECK_THROWS_AS(store.declare_series(core::RobotSeriesInfo{ .name = "one_more" }), std::invalid_argument);
    CHECK(store.series().size() == 256);
}

TEST_CASE("RobotStateStore: a schema and an encoding this build does not know are carried untouched",
          "[robot_state_store][unit]")
{
    const std::string json_schema = R"({"title":"acme.force","type":"object","properties":{"fx":{"type":"number"}}})";
    const std::string binary{ '\0', '\xff', '\x01', '\0' };

    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(concat({
        schema_record(1, "acme.force", "jsonschema", json_schema),
        channel_record(1, 1, "left_wrist_force", "json", { { "shape", "force" }, { "stale_after_ns", "500000000" } }),
        schema_record(2, "acme.Blob", "protobuf", binary),
        channel_record(2, 2, "blob", "protobuf", { { "shape", "blob" } }),
        channel_record(3, 0, "no_schema", "cbor", {}),
    }));
    store.add_samples(
        samples_payload({ .publish_seq = 1 }, concat({ message_record(1, 7, R"({"fx":1.5})"),
                                                       message_record(2, 7, binary), message_record(3, 7, "") })),
        7);

    const auto force = store.series_info("left_wrist_force");
    REQUIRE(force);
    CHECK(force->shape == "force");
    CHECK(force->schema_name == "acme.force");
    CHECK(force->schema_encoding == "jsonschema");
    CHECK(text(force->schema_data) == json_schema);
    CHECK(force->message_encoding == "json");
    CHECK(text(store.latest("left_wrist_force", 7)) == R"({"fx":1.5})");

    const auto blob = store.series_info("blob");
    REQUIRE(blob);
    CHECK(blob->schema_encoding == "protobuf");
    CHECK(text(blob->schema_data) == binary);
    CHECK(text(store.latest("blob", 7)) == binary);

    const auto no_schema = store.series_info("no_schema");
    REQUIRE(no_schema);
    CHECK(no_schema->schema_name.empty());
    CHECK(no_schema->schema_encoding.empty());
    CHECK(no_schema->schema_data.empty());
    CHECK(no_schema->message_encoding == "cbor");
    const core::RobotRecord empty = store.latest("no_schema", 7);
    REQUIRE(empty.data);
    CHECK(empty.data->empty());
    CHECK(empty.fresh);
}

TEST_CASE("RobotStateStore: declaring a series again keeps its record only when nothing changed",
          "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(robot_setup());
    store.add_samples(samples_payload({ .publish_seq = 1 },
                                      concat({ message_record(kArm, 5, "angles"), message_record(kStatus, 5, "ok") })),
                      5);
    const uint64_t setup_generation = store.setup_generation();
    const uint64_t generation = store.generation();

    store.add_setup(robot_setup());
    CHECK(store.setup_generation() == setup_generation);
    CHECK(store.generation() == generation);
    CHECK(text(store.latest("arm", 5)) == "angles");

    // The arm with another limit, under a new channel id: its place stays, its record goes.
    store.add_setup(
        channel_record(40, kArm, "arm", "flatbuffer", { { "shape", "joints" }, { "stale_after_ns", "250000000" } }));
    CHECK(store.setup_generation() == setup_generation + 1);
    CHECK(names(store) == std::vector<std::string>{ "robot_description", "arm", "status" });
    CHECK(store.series_info("arm")->stale_after_ns == 250 * kMs);
    CHECK_FALSE(store.latest("arm", 5).data);
    CHECK(text(store.latest("status", 5)) == "ok");

    // Both channel ids lead to the series, and its count of records goes on.
    store.add_samples(samples_payload({ .publish_seq = 1 }, message_record(40, 6, "by the new id")), 6);
    CHECK(text(store.latest("arm", 6)) == "by the new id");
    store.add_samples(samples_payload({ .publish_seq = 2 }, message_record(kArm, 7, "by the old id")), 7);
    CHECK(text(store.latest("arm", 7)) == "by the old id");
    CHECK(store.latest("arm", 7).sequence == 3);
}

TEST_CASE("RobotStateStore: declare_series and put_record fill a store without the wire", "[robot_state_store][unit]")
{
    core::RobotStateStore store;
    store.begin_session();
    store.declare_series(core::RobotSeriesInfo{ .name = "arm",
                                                .shape = "joints",
                                                .schema_name = "core.JointStateOutputRecord",
                                                .schema_encoding = "flatbuffer",
                                                .schema_data = { 1, 2, 3 },
                                                .message_encoding = "flatbuffer",
                                                .stale_after_ns = 100 * kMs,
                                                .in_order = false,
                                                .rate_hz = 20.0 });
    CHECK(names(store) == std::vector<std::string>{ "arm" });
    CHECK(store.series_info("arm")->schema_data == Bytes{ 1, 2, 3 });

    const Bytes angles{ 'a', 'b' };
    store.put_record("arm", angles, 1'000 * kSecond, 1'000 * kSecond + 2, 10 * kSecond,
                     core::RobotClockSample{ .robot_minus_server_ns = 990 * kSecond, .valid = true, .two_way = true });
    core::RobotRecord record = store.latest("arm", 10 * kSecond + 20 * kMs);
    CHECK(text(record) == "ab");
    CHECK(record.sequence == 1);
    CHECK(record.publish_time_ns == 1'000 * kSecond + 2);
    REQUIRE(record.age_ns);
    CHECK(*record.age_ns == 20 * kMs);
    CHECK(record.clock_two_way);
    CHECK(record.fresh);

    // Without a clock estimate there is no age; two_way alone means nothing.
    store.put_record(
        "arm", angles, 1'000 * kSecond, 1'000 * kSecond, 11 * kSecond, core::RobotClockSample{ .two_way = true });
    record = store.latest("arm", 11 * kSecond + 200 * kMs);
    CHECK(record.sequence == 2);
    CHECK_FALSE(record.age_ns);
    CHECK_FALSE(record.clock_two_way);
    CHECK(record.receive_age_ns == 200 * kMs);
    CHECK_FALSE(record.fresh);

    CHECK_THROWS_WITH(store.put_record("leg", angles, 0, 0, 0, {}), ContainsSubstring("'leg' is not declared"));
    CHECK_THROWS_AS(store.declare_series(core::RobotSeriesInfo{}), std::invalid_argument);
    CHECK_THROWS_AS(
        store.declare_series(core::RobotSeriesInfo{ .name = "x", .stale_after_ns = -1 }), std::invalid_argument);
    CHECK_THROWS_AS(store.put_record("arm", Bytes((size_t{ 1 } << 20) + 1, 0), 0, 0, 0, {}), std::invalid_argument);
    CHECK(store.latest("arm", 0).sequence == 2);
}

TEST_CASE("RobotStateStore: a snapshot carries everything into another store", "[robot_state_store][unit]")
{
    constexpr int64_t kRobotAhead = 1'000 * kSecond;

    core::RobotStateStore server;
    server.begin_session();
    server.add_setup(robot_setup());
    server.add_samples(samples_payload({ .publish_seq = 1, .robot_minus_server_ns = kRobotAhead },
                                       message_record(kDescription, 5 * kSecond + kRobotAhead, "what")),
                       5 * kSecond + kMs);
    server.add_samples(samples_payload({ .publish_seq = 1, .robot_minus_server_ns = kRobotAhead, .flags = kClockValid },
                                       message_record(kArm, 6 * kSecond + kRobotAhead, "angles")),
                       6 * kSecond + kMs);
    server.add_samples(samples_payload({ .publish_seq = 2 }, message_record(77, 1, "nobody's")), 6 * kSecond);

    core::RobotStateStore reader;
    reader.import_snapshot(server.export_snapshot());

    CHECK(reader.connected());
    CHECK(reader.session_count() == server.session_count());
    CHECK(reader.generation() == server.generation());
    CHECK(reader.setup_generation() == server.setup_generation());
    CHECK(reader.dropped_records() == 1);
    CHECK(names(reader) == names(server));
    for (const core::RobotSeriesInfo& expected : server.series())
    {
        INFO(expected.name);
        const auto info = reader.series_info(expected.name);
        REQUIRE(info);
        CHECK(info->shape == expected.shape);
        CHECK(info->schema_name == expected.schema_name);
        CHECK(info->schema_encoding == expected.schema_encoding);
        CHECK(info->schema_data == expected.schema_data);
        CHECK(info->message_encoding == expected.message_encoding);
        CHECK(info->stale_after_ns == expected.stale_after_ns);
        CHECK(info->in_order == expected.in_order);
        CHECK(info->rate_hz == expected.rate_hz);

        const int64_t now = 6 * kSecond + 40 * kMs;
        const core::RobotRecord theirs = server.latest(expected.name, now);
        const core::RobotRecord ours = reader.latest(expected.name, now);
        CHECK(text(ours) == text(theirs));
        CHECK(ours.sequence == theirs.sequence);
        CHECK(ours.sample_time_ns == theirs.sample_time_ns);
        CHECK(ours.publish_time_ns == theirs.publish_time_ns);
        CHECK(ours.receive_time_ns == theirs.receive_time_ns);
        CHECK(ours.age_ns == theirs.age_ns);
        CHECK(ours.receive_age_ns == theirs.receive_age_ns);
        CHECK(ours.clock_two_way == theirs.clock_two_way);
        CHECK(ours.fresh == theirs.fresh);
    }
    // Ages go on with the local clock.
    CHECK(*reader.latest("arm", 6 * kSecond + 40 * kMs).age_ns == 40 * kMs);
    CHECK_FALSE(reader.latest("arm", 6 * kSecond + 40 * kMs).clock_two_way);
    CHECK_FALSE(reader.latest("arm", 7 * kSecond).fresh);
    CHECK_FALSE(reader.latest("status", 7 * kSecond).data);

    // The robot app leaves, another connects: the reader follows.
    server.end_session();
    reader.import_snapshot(server.export_snapshot(reader.setup_generation()));
    CHECK_FALSE(reader.connected());
    CHECK(text(reader.latest("robot_description", 9 * kSecond)) == "what");

    server.begin_session();
    server.add_setup(channel_record(1, 0, "robot_description", "flatbuffer", { { "shape", "description" } }));
    server.add_samples(samples_payload({ .publish_seq = 1 }, message_record(1, 1, "another robot")), 1);
    reader.import_snapshot(server.export_snapshot(reader.setup_generation()));
    CHECK(reader.connected());
    CHECK(reader.session_count() == 2);
    CHECK(names(reader) == std::vector<std::string>{ "robot_description" });
    CHECK(text(reader.latest("robot_description", 1)) == "another robot");
    CHECK_FALSE(reader.latest("arm", 1).data);
}

TEST_CASE("RobotStateStore: a snapshot leaves the declarations out for a reader that has them",
          "[robot_state_store][unit]")
{
    core::RobotStateStore server;
    CHECK(server.setup_generation() == 1);
    server.begin_session();
    server.add_setup(robot_setup());
    server.add_samples(samples_payload({ .publish_seq = 1 }, message_record(kArm, 5, "angles")), 5);

    const Bytes full = server.export_snapshot();
    const Bytes brief = server.export_snapshot(server.setup_generation());
    CHECK(brief.size() < full.size());
    // 0 is no store's setup generation, so it always asks for the declarations.
    CHECK(server.export_snapshot(0) == full);
    CHECK(core::RobotStateStore().export_snapshot(0) == core::RobotStateStore().export_snapshot(7));
    CHECK(server.export_snapshot(server.setup_generation() - 1) == full);

    SECTION("a reader with the declarations takes it and keeps them")
    {
        core::RobotStateStore reader;
        reader.import_snapshot(full);
        const core::RobotRecord before = reader.latest("arm", 5);

        server.add_samples(samples_payload({ .publish_seq = 2 }, message_record(kStatus, 6, "ok")), 6);
        reader.import_snapshot(server.export_snapshot(reader.setup_generation()));
        CHECK(names(reader) == std::vector<std::string>{ "robot_description", "arm", "status" });
        CHECK(reader.series_info("arm")->schema_name == "core.JointStateOutputRecord");
        CHECK(reader.series_info("arm")->stale_after_ns == 100 * kMs);
        CHECK(text(reader.latest("status", 6)) == "ok");
        CHECK(reader.generation() == server.generation());

        // A record that did not change keeps its bytes.
        const core::RobotRecord after = reader.latest("arm", 5);
        CHECK(text(after) == "angles");
        CHECK(after.data == before.data);

        server.add_samples(samples_payload({ .publish_seq = 3 }, message_record(kArm, 7, "angles 2")), 7);
        reader.import_snapshot(server.export_snapshot(reader.setup_generation()));
        CHECK(text(reader.latest("arm", 7)) == "angles 2");
        CHECK(reader.latest("arm", 7).sequence == 2);
    }

    SECTION("a store of another setup generation refuses it and stays as it is")
    {
        core::RobotStateStore fresh;
        CHECK_THROWS_WITH(fresh.import_snapshot(brief), ContainsSubstring("setup generation"));
        CHECK(fresh.series().empty());
        CHECK(fresh.setup_generation() == 1);
        CHECK(fresh.generation() == 0);

        // A reader that is behind by one declaration.
        core::RobotStateStore reader;
        reader.import_snapshot(full);
        server.add_setup(channel_record(50, kArm, "extra", "flatbuffer", {}));
        CHECK_THROWS_AS(reader.import_snapshot(server.export_snapshot(server.setup_generation())), std::runtime_error);
        CHECK_FALSE(reader.series_info("extra"));
        reader.import_snapshot(server.export_snapshot(reader.setup_generation()));
        CHECK(reader.series_info("extra"));
        CHECK(reader.setup_generation() == server.setup_generation());
    }
}

TEST_CASE("RobotStateStore: bytes that are no snapshot of this build are refused and change nothing",
          "[robot_state_store][unit]")
{
    core::RobotStateStore server;
    server.begin_session();
    server.add_setup(robot_setup());
    server.add_samples(samples_payload({ .publish_seq = 1 }, message_record(kArm, 5, "angles")), 5);
    const Bytes good = server.export_snapshot();

    core::RobotStateStore reader;
    reader.import_snapshot(good);
    const uint64_t generation = reader.generation();

    CHECK_THROWS_AS(reader.import_snapshot(Bytes{}), std::runtime_error);
    CHECK_THROWS_AS(reader.import_snapshot(Bytes(64, 0)), std::runtime_error);
    CHECK_THROWS_AS(reader.import_snapshot(robot_setup()), std::runtime_error);

    Bytes other_version = good;
    other_version[4] ^= 0x7f;
    CHECK_THROWS_WITH(reader.import_snapshot(other_version), ContainsSubstring("not a snapshot"));

    Bytes longer = good;
    longer.push_back(0);
    CHECK_THROWS_AS(reader.import_snapshot(longer), std::runtime_error);

    // Cut anywhere, a snapshot is refused.
    for (size_t size = 0; size < good.size(); ++size)
    {
        CHECK_THROWS_AS(reader.import_snapshot(std::span<const uint8_t>(good.data(), size)), std::runtime_error);
    }

    CHECK(reader.generation() == generation);
    CHECK(reader.connected());
    CHECK(names(reader) == std::vector<std::string>{ "robot_description", "arm", "status" });
    CHECK(text(reader.latest("arm", 5)) == "angles");
}

TEST_CASE("RobotStateStore: many threads read while one writes", "[robot_state_store][unit]")
{
    constexpr int kRecords = 4'000;
    constexpr int kReaders = 4;

    core::RobotStateStore store;
    store.begin_session();
    store.add_setup(robot_setup());
    store.add_samples(samples_payload({ .publish_seq = 1 }, message_record(kDescription, 1, "what")), 1);

    std::atomic<bool> done{ false };
    std::atomic<int> failures{ 0 };
    std::atomic<uint64_t> reads{ 0 };

    // Every arm record holds its own number as text and as its sample time, so a torn read shows.
    std::vector<std::thread> readers;
    for (int reader = 0; reader < kReaders; ++reader)
    {
        readers.emplace_back(
            [&store, &done, &failures, &reads, reader]
            {
                core::RobotStateStore copy;
                uint64_t last_sequence = 0;
                uint64_t last_generation = 0;
                while (!done.load())
                {
                    const core::RobotRecord arm = store.latest("arm", 0);
                    if (arm.data && (text(arm) != std::to_string(arm.sample_time_ns) || arm.sequence < last_sequence))
                    {
                        ++failures;
                    }
                    last_sequence = arm.data ? arm.sequence : last_sequence;

                    if (text(store.latest("robot_description", 0)) != "what" || store.series().size() < 3)
                    {
                        ++failures;
                    }
                    const uint64_t generation = store.generation();
                    if (generation < last_generation || !store.connected())
                    {
                        ++failures;
                    }
                    last_generation = generation;

                    // A snapshot is one moment of the store.
                    copy.import_snapshot(store.export_snapshot(reader % 2 == 0 ? 0 : copy.setup_generation()));
                    const core::RobotRecord copied = copy.latest("arm", 0);
                    if (copied.data && text(copied) != std::to_string(copied.sample_time_ns))
                    {
                        ++failures;
                    }
                    if (copy.series().size() < 3 || text(copy.latest("robot_description", 0)) != "what")
                    {
                        ++failures;
                    }
                    ++reads;
                }
            });
    }

    for (int i = 1; i <= kRecords; ++i)
    {
        store.add_samples(
            samples_payload({ .publish_seq = static_cast<uint64_t>(i) }, message_record(kArm, i, std::to_string(i))), i);
        if (i % 500 == 0)
        {
            // A series that appears while the readers run.
            store.add_setup(channel_record(
                static_cast<uint16_t>(100 + i / 500), kArm, "late_" + std::to_string(i / 500), "flatbuffer", {}));
        }
    }
    // Let every reader see the end state at least once.
    const uint64_t reads_at_end = reads.load();
    while (reads.load() < reads_at_end + kReaders * 2)
    {
        std::this_thread::yield();
    }
    done = true;
    for (std::thread& reader : readers)
    {
        reader.join();
    }

    CHECK(failures.load() == 0);
    CHECK(reads.load() > 0);
    const core::RobotRecord arm = store.latest("arm", 0);
    CHECK(text(arm) == std::to_string(kRecords));
    CHECK(arm.sequence == static_cast<uint64_t>(kRecords));
    CHECK(store.dropped_records() == 0);
    CHECK(store.series().size() == 3 + kRecords / 500);
}
