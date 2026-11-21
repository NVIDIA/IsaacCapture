// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the robot end of robot state: RobotStatePublisher writes the MCAP Schema, Channel and Message
// records of the robot-state link message payloads itself, so every case reads them back with the MCAP library
// or compares them with what an McapWriter emits. No link and no socket is opened.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mcap/reader.hpp>
#include <mcap/sample_stream.hpp>
#include <mcap/writer.hpp>
#include <remote_link/link.hpp>
#include <robot_state_publisher/robot_state_publisher.hpp>

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using Bytes = std::vector<uint8_t>;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;

constexpr int64_t kMaxAge = 250'000'000;
constexpr size_t kRecordPrefix = 1 + sizeof(uint64_t);

Bytes bytes_of(std::string_view text)
{
    return Bytes(text.begin(), text.end());
}

// Expected bytes are spelled out field by field, little-endian, from the wire tables of the design.
void put(Bytes& out, uint64_t value, size_t width)
{
    for (size_t i = 0; i < width; ++i)
    {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void put_text(Bytes& out, std::string_view text)
{
    put(out, text.size(), 4);
    out.insert(out.end(), text.begin(), text.end());
}

Bytes joined(std::initializer_list<Bytes> parts)
{
    Bytes out;
    for (const Bytes& part : parts)
    {
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

//! Whole records (opcode, length, body) of `bytes`, which must hold nothing else.
std::vector<std::span<const uint8_t>> split_records(std::span<const uint8_t> bytes)
{
    std::vector<std::span<const uint8_t>> records;
    size_t offset = 0;
    while (offset < bytes.size())
    {
        REQUIRE(bytes.size() - offset >= kRecordPrefix);
        uint64_t body = 0;
        for (size_t i = sizeof(uint64_t); i-- > 0;)
        {
            body = (body << 8) | bytes[offset + 1 + i];
        }
        REQUIRE(body <= bytes.size() - offset - kRecordPrefix);
        records.push_back(bytes.subspan(offset, kRecordPrefix + body));
        offset += kRecordPrefix + body;
    }
    return records;
}

mcap::Record as_mcap_record(std::span<const uint8_t> record)
{
    // The Parse* functions only read through `data`; mcap::Record just does not declare it const.
    return mcap::Record{ static_cast<mcap::OpCode>(record[0]), record.size() - kRecordPrefix,
                         const_cast<std::byte*>(reinterpret_cast<const std::byte*>(record.data() + kRecordPrefix)) };
}

mcap::Schema parse_schema(std::span<const uint8_t> record)
{
    REQUIRE(static_cast<mcap::OpCode>(record[0]) == mcap::OpCode::Schema);
    mcap::Schema schema;
    REQUIRE(mcap::McapReader::ParseSchema(as_mcap_record(record), &schema).ok());
    return schema;
}

mcap::Channel parse_channel(std::span<const uint8_t> record)
{
    REQUIRE(static_cast<mcap::OpCode>(record[0]) == mcap::OpCode::Channel);
    mcap::Channel channel;
    REQUIRE(mcap::McapReader::ParseChannel(as_mcap_record(record), &channel).ok());
    return channel;
}

struct ParsedMessage
{
    uint16_t channel_id = 0;
    uint32_t sequence = 0;
    uint64_t log_time = 0;
    uint64_t publish_time = 0;
    Bytes data;
};

ParsedMessage parse_message(std::span<const uint8_t> record)
{
    REQUIRE(static_cast<mcap::OpCode>(record[0]) == mcap::OpCode::Message);
    mcap::Message message;
    REQUIRE(mcap::McapReader::ParseMessage(as_mcap_record(record), &message).ok());
    const auto* data = reinterpret_cast<const uint8_t*>(message.data);
    return ParsedMessage{ message.channelId, message.sequence, message.logTime, message.publishTime,
                          Bytes(data, data + message.dataSize) };
}

std::vector<ParsedMessage> parse_messages(std::span<const uint8_t> records)
{
    std::vector<ParsedMessage> messages;
    for (const auto record : split_records(records))
    {
        messages.push_back(parse_message(record));
    }
    return messages;
}

Bytes schema_data_of(const mcap::Schema& schema)
{
    const auto* data = reinterpret_cast<const uint8_t*>(schema.data.data());
    return Bytes(data, data + schema.data.size());
}

//! The same series and records written by the MCAP library, split the way a sample stream splits them.
struct McapReference
{
    // Declared first so it outlives the writer, which writes the end of the recording on close.
    core::TrackerSampleStreamWriter out;
    mcap::McapWriter writer;

    McapReference()
    {
        writer.open(out, core::TrackerSampleStreamWriter::writer_options());
    }

    //! Registers the series; the library writes its Schema and Channel records with the channel's first message.
    uint16_t declare(const core::RobotSeriesDeclaration& declaration, const mcap::KeyValueMap& metadata)
    {
        const auto& data = declaration.schema_data;
        mcap::Schema schema(declaration.schema_name, declaration.schema_encoding,
                            std::string_view(reinterpret_cast<const char*>(data.data()), data.size()));
        writer.addSchema(schema);
        mcap::Channel channel(declaration.name, declaration.message_encoding, schema.id, metadata);
        writer.addChannel(channel);
        return channel.id;
    }

    void write(uint16_t channel_id, uint32_t sequence, uint64_t log_time, uint64_t publish_time, const Bytes& data)
    {
        mcap::Message message;
        message.channelId = channel_id;
        message.sequence = sequence;
        message.logTime = log_time;
        message.publishTime = publish_time;
        message.data = reinterpret_cast<const std::byte*>(data.data());
        message.dataSize = data.size();
        REQUIRE(writer.write(message).ok());
    }
};

//! Stand-in for the embedded binary schema of a FlatBuffers record: bytes the publisher must not interpret.
Bytes binary_schema(uint8_t seed)
{
    Bytes data;
    for (int i = 0; i < 40; ++i)
    {
        data.push_back(static_cast<uint8_t>(seed + 7 * i));
    }
    data.push_back(0);
    return data;
}

core::RobotSeriesDeclaration description_series()
{
    core::RobotSeriesDeclaration declaration;
    declaration.name = "robot_description";
    declaration.shape = "description";
    declaration.schema_name = "core.RobotDescriptionRecord";
    declaration.schema_encoding = "flatbuffer";
    declaration.schema_data = binary_schema(1);
    declaration.message_encoding = "flatbuffer";
    declaration.stale_after_ns = 0;
    declaration.in_order = true;
    return declaration;
}

core::RobotSeriesDeclaration joints_series(std::string name)
{
    core::RobotSeriesDeclaration declaration;
    declaration.name = std::move(name);
    declaration.shape = "joints";
    declaration.schema_name = "core.JointStateOutputRecord";
    declaration.schema_encoding = "flatbuffer";
    declaration.schema_data = binary_schema(2);
    declaration.message_encoding = "flatbuffer";
    declaration.stale_after_ns = 500'000'000;
    declaration.rate_hz = 20.0;
    return declaration;
}

const std::string kForceSchema =
    R"({"$schema":"https://json-schema.org/draft/2020-12/schema","title":"two_arm_example.force","type":"object",)"
    R"("properties":{"fx":{"type":"number","unit":"N"},"fy":{"type":"number","unit":"N"}},)"
    R"("required":["fx","fy"],"additionalProperties":false})";

//! A series in a shape the robot defines: a JSON record with its JSON Schema.
core::RobotSeriesDeclaration force_series(std::string name)
{
    core::RobotSeriesDeclaration declaration;
    declaration.name = std::move(name);
    declaration.shape = "force";
    declaration.schema_name = "two_arm_example.force";
    declaration.schema_encoding = "jsonschema";
    declaration.schema_data = bytes_of(kForceSchema);
    declaration.message_encoding = "json";
    declaration.stale_after_ns = 2'000'000'000;
    declaration.rate_hz = 62.5;
    return declaration;
}

double number_of(const mcap::KeyValueMap& metadata, const std::string& key)
{
    const auto it = metadata.find(key);
    REQUIRE(it != metadata.end());
    char* end = nullptr;
    const double value = std::strtod(it->second.c_str(), &end);
    REQUIRE(end == it->second.c_str() + it->second.size());
    return value;
}

//! What a sample stream reader holds for `topic` after `setup` and each of `samples` in order.
Bytes read_latest(const Bytes& setup, std::initializer_list<Bytes> samples, std::string_view topic)
{
    core::TrackerSampleStreamReader reader;
    reader.add_setup(setup);
    uint64_t batch = 0;
    for (const Bytes& records : samples)
    {
        REQUIRE(reader.add_samples(records, ++batch, 100));
    }
    REQUIRE(reader.snapshot(100, kMaxAge));
    CHECK(reader.dropped_messages() == 0);
    const auto latest = reader.latest(topic);
    return latest.bytes ? *latest.bytes : Bytes{};
}

Bytes records_of(const Bytes& payload)
{
    REQUIRE(payload.size() >= core::kRobotSamplesPrefixSize);
    return Bytes(payload.begin() + core::kRobotSamplesPrefixSize, payload.end());
}

Bytes prefix_of(const Bytes& payload)
{
    REQUIRE(payload.size() >= core::kRobotSamplesPrefixSize);
    return Bytes(payload.begin(), payload.begin() + core::kRobotSamplesPrefixSize);
}

core::RobotSamplesPrefix prefix(uint64_t publish_seq)
{
    core::RobotSamplesPrefix prefix;
    prefix.publish_seq = publish_seq;
    prefix.robot_send_time_ns = 1'000 * static_cast<int64_t>(publish_seq);
    prefix.robot_minus_server_ns = -5'000;
    prefix.clock_valid = true;
    return prefix;
}

} // namespace

TEST_CASE("RobotStatePublisher: Schema, Channel and Message records are the bytes of the wire tables",
          "[robot_state][publisher]")
{
    core::RobotSeriesDeclaration declaration;
    declaration.name = "a";
    declaration.shape = "s";
    declaration.schema_name = "n";
    declaration.schema_encoding = "e";
    declaration.schema_data = { 0xAA, 0xBB };
    declaration.message_encoding = "m";
    declaration.stale_after_ns = 7;

    core::RobotStatePublisher publisher;
    publisher.declare_series(declaration);
    const Bytes setup = publisher.take_setup();
    const auto records = split_records(setup);
    REQUIRE(records.size() == 2);

    Bytes schema;
    put(schema, 0x03, 1);
    put(schema, 18, 8); // body length
    put(schema, 1, 2); // id
    put_text(schema, "n");
    put_text(schema, "e");
    put(schema, 2, 4);
    put(schema, 0xAA, 1);
    put(schema, 0xBB, 1);
    CHECK(Bytes(records[0].begin(), records[0].end()) == schema);

    // The text of rate_hz is free; every other byte is fixed.
    const std::string rate_text = parse_channel(records[1]).metadata.at("rate_hz");
    CHECK(number_of(parse_channel(records[1]).metadata, "rate_hz") == 0.0);
    const size_t metadata_size = (4 + 8 + 4 + 6) + (4 + 7 + 4 + rate_text.size()) + (4 + 5 + 4 + 1) + (4 + 14 + 4 + 1);
    Bytes channel;
    put(channel, 0x04, 1);
    put(channel, 2 + 2 + (4 + 1) + (4 + 1) + 4 + metadata_size, 8); // body length
    put(channel, 1, 2); // id
    put(channel, 1, 2); // schema id
    put_text(channel, "a");
    put_text(channel, "m");
    put(channel, metadata_size, 4);
    put_text(channel, "delivery");
    put_text(channel, "latest");
    put_text(channel, "rate_hz");
    put_text(channel, rate_text);
    put_text(channel, "shape");
    put_text(channel, "s");
    put_text(channel, "stale_after_ns");
    put_text(channel, "7");
    CHECK(Bytes(records[1].begin(), records[1].end()) == channel);

    publisher.publish("a", Bytes{ 0x10, 0x20, 0x30 }, 0x0102030405060708, 0x1112131415161718);
    Bytes message;
    put(message, 0x05, 1);
    put(message, 2 + 4 + 8 + 8 + 3, 8); // body length
    put(message, 1, 2); // channel id
    put(message, 1, 4); // sequence
    put(message, 0x0102030405060708, 8); // log time: the sample time
    put(message, 0x1112131415161718, 8); // publish time
    put(message, 0x10, 1);
    put(message, 0x20, 1);
    put(message, 0x30, 1);
    CHECK(publisher.take_latest_records() == message);
}

TEST_CASE("RobotStatePublisher: setup and records equal what the MCAP library writes", "[robot_state][publisher]")
{
    const std::vector<core::RobotSeriesDeclaration> declarations = { description_series(), joints_series("arm"),
                                                                     force_series("left_wrist_force") };
    core::RobotStatePublisher publisher;
    for (const auto& declaration : declarations)
    {
        publisher.declare_series(declaration);
    }
    const Bytes setup = publisher.take_setup();
    const auto setup_records = split_records(setup);
    REQUIRE(setup_records.size() == 2 * declarations.size());

    const Bytes description = bytes_of("a description record");
    const Bytes arm = bytes_of("an arm record");
    const Bytes force = bytes_of(R"({"fx":1.5,"fy":-2})");
    publisher.publish("robot_description", description, 10, 11);
    publisher.publish("arm", arm, 20, 21);
    publisher.publish("left_wrist_force", force, 30, 31);

    McapReference reference;
    for (size_t i = 0; i < declarations.size(); ++i)
    {
        // The publisher's own metadata, read by the library, so the comparison does not fix the text of a number.
        const mcap::Channel channel = parse_channel(setup_records[2 * i + 1]);
        CHECK(reference.declare(declarations[i], channel.metadata) == i + 1);
    }
    reference.write(1, 1, 10, 11, description);
    reference.write(2, 1, 20, 21, arm);
    reference.write(3, 1, 30, 31, force);

    CHECK(setup == reference.out.take_setup());
    const Bytes records = joined({ publisher.take_in_order_records(), publisher.take_latest_records() });
    CHECK(records == reference.out.take_samples());
}

TEST_CASE("RobotStatePublisher: setup declares each series with its metadata", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    publisher.declare_series(description_series());
    publisher.declare_series(joints_series("arm"));
    publisher.declare_series(force_series("left_wrist_force"));
    const Bytes setup = publisher.take_setup();
    const auto records = split_records(setup);
    REQUIRE(records.size() == 6);

    const mcap::Schema description_schema = parse_schema(records[0]);
    CHECK(description_schema.id == 1);
    CHECK(description_schema.name == "core.RobotDescriptionRecord");
    CHECK(description_schema.encoding == "flatbuffer");
    CHECK(schema_data_of(description_schema) == binary_schema(1));

    const mcap::Channel description = parse_channel(records[1]);
    CHECK(description.id == 1);
    CHECK(description.schemaId == 1);
    CHECK(description.topic == "robot_description");
    CHECK(description.messageEncoding == "flatbuffer");
    CHECK(description.metadata.size() == 4);
    CHECK(description.metadata.at("shape") == "description");
    CHECK(description.metadata.at("delivery") == "in_order");
    CHECK(number_of(description.metadata, "stale_after_ns") == 0.0);
    CHECK(number_of(description.metadata, "rate_hz") == 0.0);

    CHECK(parse_schema(records[2]).id == 2);
    const mcap::Channel arm = parse_channel(records[3]);
    CHECK(arm.id == 2);
    CHECK(arm.schemaId == 2);
    CHECK(arm.topic == "arm");
    CHECK(arm.metadata.at("shape") == "joints");
    CHECK(arm.metadata.at("delivery") == "latest");
    CHECK(arm.metadata.at("stale_after_ns") == "500000000");
    CHECK(number_of(arm.metadata, "rate_hz") == 20.0);

    const mcap::Channel force = parse_channel(records[5]);
    CHECK(force.id == 3);
    CHECK(force.metadata.at("stale_after_ns") == "2000000000");
    CHECK(number_of(force.metadata, "rate_hz") == 62.5);

    // The library's stream reader takes the same bytes.
    core::TrackerSampleStreamReader reader;
    reader.add_setup(setup);
    REQUIRE(reader.snapshot(0, kMaxAge) == false);
    for (const char* topic : { "robot_description", "arm", "left_wrist_force" })
    {
        REQUIRE(reader.channel(topic) != nullptr);
        CHECK(reader.channel(topic)->schema != nullptr);
    }
    CHECK(reader.channel("arm")->message_encoding == "flatbuffer");
    CHECK(reader.channel("arm")->schema->name == "core.JointStateOutputRecord");
}

TEST_CASE("RobotStatePublisher: a rate keeps its value through the metadata text", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    const double rates[] = { 0.0, 1.0 / 3.0, 20.0, 0.1, 1e-9, 123456789.125, std::numeric_limits<double>::max() };
    size_t index = 0;
    for (const double rate : rates)
    {
        auto declaration = joints_series("series" + std::to_string(index++));
        declaration.rate_hz = rate;
        declaration.stale_after_ns = std::numeric_limits<int64_t>::max();
        publisher.declare_series(declaration);
    }
    const Bytes setup = publisher.take_setup();
    const auto records = split_records(setup);
    REQUIRE(records.size() == 2 * std::size(rates));
    for (size_t i = 0; i < std::size(rates); ++i)
    {
        const mcap::Channel channel = parse_channel(records[2 * i + 1]);
        CHECK(number_of(channel.metadata, "rate_hz") == rates[i]);
        CHECK(channel.metadata.at("stale_after_ns") == "9223372036854775807");
    }
}

TEST_CASE("RobotStatePublisher: take_setup holds the series declared since the last call", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    CHECK(publisher.take_setup().empty());
    CHECK_FALSE(publisher.has_series("arm"));

    publisher.declare_series(description_series());
    publisher.declare_series(joints_series("arm"));
    CHECK(publisher.has_series("arm"));
    CHECK(publisher.has_series("robot_description"));
    CHECK(publisher.in_order("robot_description"));
    CHECK_FALSE(publisher.in_order("arm"));
    CHECK(split_records(publisher.take_setup()).size() == 4);
    CHECK(publisher.take_setup().empty());

    publisher.declare_series(joints_series("waist"));
    const Bytes later = publisher.take_setup();
    const auto records = split_records(later);
    REQUIRE(records.size() == 2);
    CHECK(parse_schema(records[0]).id == 3);
    CHECK(parse_channel(records[1]).id == 3);
    CHECK(parse_channel(records[1]).topic == "waist");
    CHECK(publisher.take_setup().empty());
}

TEST_CASE("RobotStatePublisher: a newest-only series keeps one record until taken", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    publisher.declare_series(description_series());
    publisher.declare_series(joints_series("arm"));
    const Bytes setup = publisher.take_setup();
    CHECK(publisher.take_latest_records().empty());

    publisher.publish("arm", bytes_of("first"), 100, 101);
    publisher.publish("arm", bytes_of("second"), 200, 201);
    publisher.publish("arm", bytes_of("third"), 300, 301);
    CHECK(publisher.take_in_order_records().empty());

    const Bytes records = publisher.take_latest_records();
    const auto messages = parse_messages(records);
    REQUIRE(messages.size() == 1);
    CHECK(messages[0].channel_id == 2);
    CHECK(messages[0].sequence == 3);
    CHECK(messages[0].log_time == 300);
    CHECK(messages[0].publish_time == 301);
    CHECK(messages[0].data == bytes_of("third"));
    CHECK(publisher.take_latest_records().empty());
    CHECK(read_latest(setup, { records }, "arm") == bytes_of("third"));

    // The counter of the series goes on after a take.
    publisher.publish("arm", bytes_of("fourth"), 400, 401);
    const auto next = parse_messages(publisher.take_latest_records());
    REQUIRE(next.size() == 1);
    CHECK(next[0].sequence == 4);
    CHECK(next[0].data == bytes_of("fourth"));
}

TEST_CASE("RobotStatePublisher: newest-only records leave in the order of their publish calls", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    publisher.declare_series(joints_series("left_arm"));
    publisher.declare_series(joints_series("right_arm"));
    publisher.declare_series(force_series("left_wrist_force"));
    publisher.take_setup();

    publisher.publish("right_arm", bytes_of("r1"), 1, 1);
    publisher.publish("left_wrist_force", bytes_of("f1"), 2, 2);
    publisher.publish("left_arm", bytes_of("l1"), 3, 3);
    publisher.publish("right_arm", bytes_of("r2"), 4, 4);

    const auto messages = parse_messages(publisher.take_latest_records());
    REQUIRE(messages.size() == 3);
    CHECK(messages[0].channel_id == 3);
    CHECK(messages[0].data == bytes_of("f1"));
    CHECK(messages[1].channel_id == 1);
    CHECK(messages[1].data == bytes_of("l1"));
    CHECK(messages[2].channel_id == 2);
    CHECK(messages[2].sequence == 2);
    CHECK(messages[2].data == bytes_of("r2"));
}

TEST_CASE("RobotStatePublisher: an in-order series keeps every record, in publish order", "[robot_state][publisher]")
{
    core::RobotSeriesDeclaration events = description_series();
    events.name = "events";
    events.shape = "status";

    core::RobotStatePublisher publisher;
    publisher.declare_series(description_series());
    publisher.declare_series(joints_series("arm"));
    publisher.declare_series(events);
    const Bytes setup = publisher.take_setup();
    CHECK(publisher.take_in_order_records().empty());

    publisher.publish("robot_description", bytes_of("d1"), 10, 11);
    publisher.publish("events", bytes_of("e1"), 20, 21);
    publisher.publish("arm", bytes_of("a1"), 25, 26);
    publisher.publish("robot_description", bytes_of("d2"), 30, 31);
    publisher.publish("events", bytes_of("e2"), 40, 41);

    const Bytes records = publisher.take_in_order_records();
    const auto messages = parse_messages(records);
    REQUIRE(messages.size() == 4);
    const uint16_t channels[] = { 1, 3, 1, 3 };
    const uint32_t sequences[] = { 1, 1, 2, 2 };
    const uint64_t log_times[] = { 10, 20, 30, 40 };
    const char* data[] = { "d1", "e1", "d2", "e2" };
    for (size_t i = 0; i < messages.size(); ++i)
    {
        CHECK(messages[i].channel_id == channels[i]);
        CHECK(messages[i].sequence == sequences[i]);
        CHECK(messages[i].log_time == log_times[i]);
        CHECK(messages[i].publish_time == log_times[i] + 1);
        CHECK(messages[i].data == bytes_of(data[i]));
    }
    CHECK(publisher.take_in_order_records().empty());
    CHECK(parse_messages(publisher.take_latest_records()).size() == 1);
    CHECK(read_latest(setup, { records }, "robot_description") == bytes_of("d2"));
}

TEST_CASE("RobotStatePublisher: times before zero and an empty record are carried as given", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    publisher.declare_series(joints_series("arm"));
    publisher.take_setup();

    publisher.publish("arm", Bytes{}, -1, std::numeric_limits<int64_t>::max());
    const auto messages = parse_messages(publisher.take_latest_records());
    REQUIRE(messages.size() == 1);
    CHECK(static_cast<int64_t>(messages[0].log_time) == -1);
    CHECK(messages[0].publish_time == static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    CHECK(messages[0].data.empty());
}

TEST_CASE("RobotStatePublisher: a robot's own shape travels with its description", "[robot_state][publisher]")
{
    // A shape this build has no code for: its schema name, encodings and schema bytes are carried untouched.
    core::RobotSeriesDeclaration foreign;
    foreign.name = "tactile";
    foreign.shape = "skin";
    foreign.schema_name = "vendor.SkinPatch";
    foreign.schema_encoding = "protobuf";
    foreign.schema_data = { 0x00, 0xFF, 0x0A, 0x00, 0x80 };
    foreign.message_encoding = "protobuf";

    core::RobotStatePublisher publisher;
    publisher.declare_series(force_series("left_wrist_force"));
    publisher.declare_series(foreign);
    const Bytes setup = publisher.take_setup();
    const auto records = split_records(setup);
    REQUIRE(records.size() == 4);

    const mcap::Schema force_schema = parse_schema(records[0]);
    CHECK(force_schema.name == "two_arm_example.force");
    CHECK(force_schema.encoding == "jsonschema");
    CHECK(schema_data_of(force_schema) == bytes_of(kForceSchema));
    const mcap::Channel force_channel = parse_channel(records[1]);
    CHECK(force_channel.topic == "left_wrist_force");
    CHECK(force_channel.messageEncoding == "json");
    CHECK(force_channel.metadata.at("shape") == "force");

    const mcap::Schema foreign_schema = parse_schema(records[2]);
    CHECK(foreign_schema.name == "vendor.SkinPatch");
    CHECK(foreign_schema.encoding == "protobuf");
    CHECK(schema_data_of(foreign_schema) == foreign.schema_data);
    CHECK(parse_channel(records[3]).messageEncoding == "protobuf");
    CHECK(parse_channel(records[3]).metadata.at("shape") == "skin");

    const Bytes force = bytes_of(R"({"fx":1.5,"fy":-2})");
    const Bytes patch = { 0x08, 0x00, 0x12, 0x00 };
    publisher.publish("left_wrist_force", force, 1, 2);
    publisher.publish("tactile", patch, 3, 4);
    const Bytes samples = publisher.take_latest_records();

    // A reader that has only these bytes finds the schema beside each record.
    core::TrackerSampleStreamReader reader;
    reader.add_setup(setup);
    REQUIRE(reader.add_samples(samples, 1, 100));
    REQUIRE(reader.snapshot(100, kMaxAge));
    REQUIRE(reader.latest("left_wrist_force").bytes);
    CHECK(*reader.latest("left_wrist_force").bytes == force);
    REQUIRE(reader.channel("left_wrist_force") != nullptr);
    REQUIRE(reader.channel("left_wrist_force")->schema != nullptr);
    CHECK(schema_data_of(*reader.channel("left_wrist_force")->schema) == bytes_of(kForceSchema));
    REQUIRE(reader.latest("tactile").bytes);
    CHECK(*reader.latest("tactile").bytes == patch);
    CHECK(reader.channel("tactile")->message_encoding == "protobuf");
}

TEST_CASE("RobotStatePublisher: a series without a schema is declared with an empty one", "[robot_state][publisher]")
{
    core::RobotSeriesDeclaration bare;
    bare.name = "notes";
    bare.message_encoding = "json";

    core::RobotStatePublisher publisher;
    publisher.declare_series(bare);
    const Bytes setup = publisher.take_setup();
    const auto records = split_records(setup);
    REQUIRE(records.size() == 2);
    const mcap::Schema schema = parse_schema(records[0]);
    CHECK(schema.id == 1);
    CHECK(schema.name.empty());
    CHECK(schema.encoding.empty());
    CHECK(schema.data.empty());
    const mcap::Channel channel = parse_channel(records[1]);
    CHECK(channel.schemaId == 1);
    CHECK(channel.metadata.at("shape").empty());
}

TEST_CASE("RobotStatePublisher: the samples prefix is 32 little-endian bytes", "[robot_state][publisher]")
{
    core::RobotSamplesPrefix prefix;
    prefix.publish_seq = 0x0102030405060708;
    prefix.robot_send_time_ns = 0x1112131415161718;
    prefix.robot_minus_server_ns = -2;
    prefix.clock_valid = true;
    prefix.clock_two_way = true;
    const Bytes records = { 0xDE, 0xAD };

    const Bytes expected = {
        0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // publish_seq
        0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, // robot_send_time_ns
        0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // robot_minus_server_ns
        0x03, 0x00, 0x00, 0x00, // flags: clock valid, clock two-way
        0x00, 0x00, 0x00, 0x00, // reserved
        0xDE, 0xAD,
    };
    CHECK(core::kRobotSamplesPrefixSize == 32);
    CHECK(core::encode_robot_samples_payload(prefix, records) == expected);

    prefix.clock_two_way = false;
    CHECK(core::encode_robot_samples_payload(prefix, records)[24] == 0x01);
    prefix.clock_valid = false;
    prefix.clock_two_way = true;
    CHECK(core::encode_robot_samples_payload(prefix, records)[24] == 0x02);

    const Bytes empty = core::encode_robot_samples_payload(core::RobotSamplesPrefix{}, {});
    CHECK(empty == Bytes(32, 0));
}

TEST_CASE("RobotStatePublisher: merge keeps the newer prefix and the last record of each channel",
          "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    publisher.declare_series(joints_series("left_arm"));
    publisher.declare_series(joints_series("right_arm"));
    publisher.declare_series(joints_series("waist"));
    const Bytes setup = publisher.take_setup();

    const auto one_record = [&publisher](std::string_view series, std::string_view data, int64_t time_ns)
    {
        publisher.publish(series, bytes_of(data), time_ns, time_ns);
        return publisher.take_latest_records();
    };
    const Bytes left1 = one_record("left_arm", "l1", 1);
    const Bytes right1 = one_record("right_arm", "r1", 2);
    const Bytes left2 = one_record("left_arm", "l2", 3);
    const Bytes waist1 = one_record("waist", "w1", 4);

    const Bytes older = core::encode_robot_samples_payload(prefix(1), joined({ left1, right1 }));
    const Bytes newer = core::encode_robot_samples_payload(prefix(2), joined({ left2, waist1 }));

    // Usable as the merge function of Link::send_latest().
    const core::MergeFn merge = core::merge_robot_samples_payloads;
    const Bytes merged = merge(older, newer);
    CHECK(prefix_of(merged) == prefix_of(newer));
    CHECK(records_of(merged) == joined({ right1, left2, waist1 }));

    // It reads as the two payloads read in order.
    for (const char* topic : { "left_arm", "right_arm", "waist" })
    {
        CHECK(read_latest(setup, { records_of(merged) }, topic) ==
              read_latest(setup, { records_of(older), records_of(newer) }, topic));
    }
    CHECK(read_latest(setup, { records_of(merged) }, "left_arm") == bytes_of("l2"));
    CHECK(read_latest(setup, { records_of(merged) }, "right_arm") == bytes_of("r1"));

    // A third unsent payload merges into the result the same way.
    const Bytes right2 = one_record("right_arm", "r2", 5);
    const Bytes third = core::encode_robot_samples_payload(prefix(3), right2);
    const Bytes all = core::merge_robot_samples_payloads(merged, third);
    CHECK(prefix_of(all) == prefix_of(third));
    CHECK(records_of(all) == joined({ left2, waist1, right2 }));
}

TEST_CASE("RobotStatePublisher: merge of payloads without records is the newer prefix", "[robot_state][publisher]")
{
    const Bytes older = core::encode_robot_samples_payload(prefix(1), {});
    const Bytes newer = core::encode_robot_samples_payload(prefix(2), {});
    CHECK(core::merge_robot_samples_payloads(older, newer) == newer);

    core::RobotStatePublisher publisher;
    publisher.declare_series(joints_series("arm"));
    publisher.publish("arm", bytes_of("a1"), 1, 1);
    const Bytes with_record = core::encode_robot_samples_payload(prefix(1), publisher.take_latest_records());
    CHECK(records_of(core::merge_robot_samples_payloads(with_record, newer)) == records_of(with_record));
    CHECK(prefix_of(core::merge_robot_samples_payloads(with_record, newer)) == newer);
    CHECK(core::merge_robot_samples_payloads(older, with_record) == with_record);
}

TEST_CASE("RobotStatePublisher: merge refuses a malformed payload", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    publisher.declare_series(joints_series("arm"));
    const Bytes setup = publisher.take_setup();
    publisher.publish("arm", bytes_of("a1"), 1, 1);
    const Bytes record = publisher.take_latest_records();
    const Bytes good = core::encode_robot_samples_payload(prefix(1), record);

    const Bytes short_prefix(core::kRobotSamplesPrefixSize - 1, 0);
    Bytes cut = good;
    cut.pop_back();
    Bytes cut_in_length = good;
    cut_in_length.resize(core::kRobotSamplesPrefixSize + 4);
    const Bytes not_a_message = core::encode_robot_samples_payload(prefix(1), setup);
    Bytes headless;
    put(headless, 0x05, 1);
    put(headless, 21, 8);
    headless.resize(headless.size() + 21, 0);
    const Bytes short_message = core::encode_robot_samples_payload(prefix(1), headless);

    for (const Bytes& bad : { Bytes{}, short_prefix, cut, cut_in_length, not_a_message, short_message })
    {
        CHECK_THROWS_AS(core::merge_robot_samples_payloads(bad, good), std::runtime_error);
        CHECK_THROWS_AS(core::merge_robot_samples_payloads(good, bad), std::runtime_error);
    }
    using core::merge_robot_samples_payloads;
    CHECK_THROWS_WITH(merge_robot_samples_payloads(short_prefix, good), ContainsSubstring("shorter than its prefix"));
    CHECK_THROWS_WITH(merge_robot_samples_payloads(good, cut), ContainsSubstring("end inside a record"));
    CHECK_THROWS_WITH(merge_robot_samples_payloads(not_a_message, good), ContainsSubstring("other than a Message"));
    CHECK_NOTHROW(core::merge_robot_samples_payloads(good, good));
}

TEST_CASE("RobotStatePublisher: refuses a bad declaration", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    publisher.declare_series(joints_series("arm"));
    publisher.take_setup();

    core::RobotSeriesDeclaration unnamed = joints_series("");
    CHECK_THROWS_AS(publisher.declare_series(unnamed), std::invalid_argument);
    CHECK_THROWS_WITH(publisher.declare_series(unnamed), StartsWith("RobotStatePublisher: "));

    CHECK_THROWS_AS(publisher.declare_series(joints_series("arm")), std::invalid_argument);
    CHECK_THROWS_WITH(publisher.declare_series(force_series("arm")), StartsWith("RobotStatePublisher: ") &&
                                                                         ContainsSubstring("'arm'") &&
                                                                         ContainsSubstring("declared twice"));

    core::RobotSeriesDeclaration negative_age = joints_series("waist");
    negative_age.stale_after_ns = -1;
    CHECK_THROWS_AS(publisher.declare_series(negative_age), std::invalid_argument);
    CHECK_THROWS_WITH(
        publisher.declare_series(negative_age), StartsWith("RobotStatePublisher: ") && ContainsSubstring("'waist'"));

    using Limits = std::numeric_limits<double>;
    for (const double rate : { -1.0, Limits::infinity(), Limits::quiet_NaN() })
    {
        core::RobotSeriesDeclaration bad_rate = joints_series("waist");
        bad_rate.rate_hz = rate;
        CHECK_THROWS_AS(publisher.declare_series(bad_rate), std::invalid_argument);
        CHECK_THROWS_WITH(
            publisher.declare_series(bad_rate), StartsWith("RobotStatePublisher: ") && ContainsSubstring("'waist'"));
    }

    // A refused declaration declares nothing.
    CHECK_FALSE(publisher.has_series("waist"));
    CHECK(publisher.take_setup().empty());
    publisher.declare_series(joints_series("waist"));
    const Bytes setup = publisher.take_setup();
    const auto records = split_records(setup);
    REQUIRE(records.size() == 2);
    CHECK(parse_channel(records[1]).id == 2);
}

TEST_CASE("RobotStatePublisher: refuses more series than the limit", "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    for (size_t i = 0; i < core::kMaxRobotSeries; ++i)
    {
        publisher.declare_series(joints_series("series" + std::to_string(i)));
    }
    const Bytes setup = publisher.take_setup();
    const auto records = split_records(setup);
    REQUIRE(records.size() == 2 * core::kMaxRobotSeries);
    CHECK(parse_channel(records.back()).id == core::kMaxRobotSeries);
    CHECK(parse_channel(records.back()).topic == "series255");

    CHECK_THROWS_AS(publisher.declare_series(joints_series("one_more")), std::invalid_argument);
    CHECK_THROWS_WITH(publisher.declare_series(joints_series("one_more")),
                      StartsWith("RobotStatePublisher: ") && ContainsSubstring("'one_more'") && ContainsSubstring("256"));
    CHECK_FALSE(publisher.has_series("one_more"));
    CHECK(publisher.take_setup().empty());
}

TEST_CASE("RobotStatePublisher: refuses a record for an undeclared series or above the size limit",
          "[robot_state][publisher]")
{
    core::RobotStatePublisher publisher;
    publisher.declare_series(description_series());
    publisher.declare_series(joints_series("arm"));
    publisher.take_setup();

    CHECK_THROWS_AS(publisher.publish("leg", bytes_of("x"), 1, 1), std::invalid_argument);
    CHECK_THROWS_WITH(publisher.publish("leg", bytes_of("x"), 1, 1), StartsWith("RobotStatePublisher: ") &&
                                                                         ContainsSubstring("'leg'") &&
                                                                         ContainsSubstring("not declared"));
    CHECK_THROWS_AS(publisher.in_order("leg"), std::invalid_argument);
    CHECK_THROWS_WITH(publisher.in_order("leg"), StartsWith("RobotStatePublisher: ") && ContainsSubstring("'leg'"));

    const Bytes too_large(core::kMaxRobotRecordBytes + 1, 0x55);
    for (const char* series : { "arm", "robot_description" })
    {
        CHECK_THROWS_AS(publisher.publish(series, too_large, 1, 1), std::invalid_argument);
        CHECK_THROWS_WITH(publisher.publish(series, too_large, 1, 1), StartsWith("RobotStatePublisher: ") &&
                                                                          ContainsSubstring(series) &&
                                                                          ContainsSubstring("1048577"));
    }

    // A refused record leaves nothing to send and takes no sequence number.
    CHECK(publisher.take_latest_records().empty());
    CHECK(publisher.take_in_order_records().empty());

    const Bytes largest(core::kMaxRobotRecordBytes, 0x55);
    publisher.publish("arm", largest, 1, 1);
    publisher.publish("robot_description", largest, 2, 2);
    const auto latest = parse_messages(publisher.take_latest_records());
    REQUIRE(latest.size() == 1);
    CHECK(latest[0].sequence == 1);
    CHECK(latest[0].data == largest);
    const auto in_order = parse_messages(publisher.take_in_order_records());
    REQUIRE(in_order.size() == 1);
    CHECK(in_order[0].sequence == 1);
    CHECK(in_order[0].data.size() == core::kMaxRobotRecordBytes);
}
