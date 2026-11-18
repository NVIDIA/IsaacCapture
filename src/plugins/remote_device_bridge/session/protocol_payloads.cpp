// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/protocol_payloads.hpp"

#include "protocol_io.hpp"

#include <string>

namespace remote_plugin_protocol
{

using detail::build_frame;
using detail::Reader;
using detail::Writer;

namespace
{

uint8_t to_wire_hand(XrHandEXT hand)
{
    switch (hand)
    {
    case XR_HAND_LEFT_EXT:
        return 0;
    case XR_HAND_RIGHT_EXT:
        return 1;
    default:
        throw std::invalid_argument("protocol: unsupported XrHandEXT value");
    }
}

XrHandEXT from_wire_hand(uint8_t hand)
{
    switch (hand)
    {
    case 0:
        return XR_HAND_LEFT_EXT;
    case 1:
        return XR_HAND_RIGHT_EXT;
    default:
        throw DecodeError("protocol: invalid hand value " + std::to_string(hand));
    }
}

uint8_t to_wire_is_left(XrHandEXT hand)
{
    switch (hand)
    {
    case XR_HAND_LEFT_EXT:
        return 1;
    case XR_HAND_RIGHT_EXT:
        return 0;
    default:
        throw std::invalid_argument("protocol: unsupported XrHandEXT value");
    }
}

uint8_t to_wire_mode(core::WristTrackingSourceMode mode)
{
    switch (mode)
    {
    case core::WristTrackingSourceMode::Auto:
        return 0;
    case core::WristTrackingSourceMode::HandTracking:
        return 1;
    case core::WristTrackingSourceMode::Controller:
        return 2;
    }
    throw std::invalid_argument("protocol: unsupported wrist tracking source mode");
}

core::WristTrackingSourceMode from_wire_mode(uint8_t mode)
{
    switch (mode)
    {
    case 0:
        return core::WristTrackingSourceMode::Auto;
    case 1:
        return core::WristTrackingSourceMode::HandTracking;
    case 2:
        return core::WristTrackingSourceMode::Controller;
    default:
        throw DecodeError("protocol: invalid wrist tracking source mode " + std::to_string(mode));
    }
}

void write_wrist_config(Writer& writer, const core::WristTrackingSourceConfig& value)
{
    writer.u8(to_wire_mode(value.mode));
    writer.pose(value.left_aim_to_wrist);
    writer.pose(value.right_aim_to_wrist);
}

core::WristTrackingSourceConfig read_wrist_config(Reader& reader)
{
    core::WristTrackingSourceConfig value;
    value.mode = from_wire_mode(reader.u8());
    value.left_aim_to_wrist = reader.pose();
    value.right_aim_to_wrist = reader.pose();
    return value;
}

//! Runs `write` against a fresh payload buffer and returns it.
template <typename Write>
std::vector<uint8_t> build_payload(Write write)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    write(writer);
    return out;
}

//! Runs `read` over `payload` and rejects trailing bytes; the payload-side mirror of build_payload.
template <typename Read>
auto parse_payload(std::span<const uint8_t> payload, const char* what, Read read)
{
    Reader reader(payload, what);
    auto value = read(reader);
    reader.finish();
    return value;
}

} // namespace

std::vector<uint8_t> encode_hand_sample_frame(uint16_t channel_id, HandJoints joints, int64_t sample_time_ns)
{
    return build_frame(FrameType::HandSample, channel_id, kHandSamplePayloadSize,
                       [&](Writer& writer)
                       {
                           writer.i64(sample_time_ns);
                           for (const XrHandJointLocationEXT& joint : joints)
                           {
                               writer.u64(joint.locationFlags);
                               writer.pose(joint.pose);
                               writer.f32(joint.radius);
                           }
                       });
}

std::vector<uint8_t> encode_schema_sample_frame(uint16_t channel_id,
                                                std::span<const uint8_t> bytes,
                                                int64_t sample_time_local_common_clock_ns,
                                                int64_t sample_time_raw_device_clock_ns)
{
    return build_frame(FrameType::SchemaSample, channel_id, kSchemaSampleFixedSize + bytes.size(),
                       [&](Writer& writer)
                       {
                           writer.i64(sample_time_local_common_clock_ns);
                           writer.i64(sample_time_raw_device_clock_ns);
                           writer.u32(static_cast<uint32_t>(bytes.size()));
                           writer.bytes(bytes);
                       });
}

std::vector<uint8_t> encode_hello(const Hello& value)
{
    return build_payload(
        [&](Writer& writer)
        {
            writer.u16(kHelloVersion);
            writer.str(value.app_name);
            writer.boolean(value.requirements.schema_push);
            writer.boolean(value.requirements.hand_tracking_push);
            writer.boolean(value.requirements.wrist_tracking_pull.has_value());
            if (value.requirements.wrist_tracking_pull)
            {
                write_wrist_config(writer, *value.requirements.wrist_tracking_pull);
            }
        });
}

std::vector<uint8_t> encode_hello_ack(const HelloAck& value)
{
    return build_payload(
        [&](Writer& writer)
        {
            writer.boolean(value.accepted);
            writer.str(value.reason);
        });
}

std::vector<uint8_t> encode_open_hand(XrHandEXT hand)
{
    return build_payload([&](Writer& writer) { writer.u8(to_wire_hand(hand)); });
}

std::vector<uint8_t> encode_open_schema(const core::SchemaPusherConfig& value)
{
    return build_payload(
        [&](Writer& writer)
        {
            writer.str(value.collection_id);
            writer.u64(static_cast<uint64_t>(value.max_flatbuffer_size));
            writer.str(value.tensor_identifier);
            writer.str(value.localized_name);
            writer.str(value.app_name);
        });
}

std::vector<uint8_t> encode_wrist_sample(const WristSample& value)
{
    return build_payload(
        [&](Writer& writer)
        {
            writer.i64(value.sample_time_ns);
            writer.u8(to_wire_is_left(value.hand));
            writer.boolean(value.sample.valid);
            writer.boolean(value.sample.tracked);
            writer.pose(value.sample.pose);
        });
}

std::vector<uint8_t> encode_error(const std::string& message)
{
    return build_payload([&](Writer& writer) { writer.str(message); });
}

std::vector<uint8_t> encode_ping(int64_t t1_ns)
{
    return build_payload([&](Writer& writer) { writer.i64(t1_ns); });
}

std::vector<uint8_t> encode_pong(const Pong& value)
{
    return build_payload(
        [&](Writer& writer)
        {
            writer.i64(value.ping_sent_ns);
            writer.i64(value.ping_received_ns);
            writer.i64(value.pong_sent_ns);
        });
}

void set_pong_sent_time(std::vector<uint8_t>& pong_frame, int64_t pong_sent_ns)
{
    // PONG payload: ping_sent, ping_received, pong_sent (i64 each).
    constexpr size_t kPongSentOffset = kHeaderSize + 2 * sizeof(int64_t);
    if (pong_frame.size() != kHeaderSize + 3 * sizeof(int64_t) || decode_header(pong_frame).type != FrameType::Pong)
    {
        throw std::invalid_argument("protocol: not an encoded PONG frame");
    }
    detail::store_le(pong_frame.data() + kPongSentOffset, pong_sent_ns);
}

Hello decode_hello(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "HELLO",
                         [](Reader& reader)
                         {
                             Hello value;
                             if (const uint16_t version = reader.u16(); version != kHelloVersion)
                             {
                                 throw DecodeError("protocol: unsupported HELLO version " + std::to_string(version));
                             }
                             value.app_name = reader.str();
                             value.requirements.schema_push = reader.boolean();
                             value.requirements.hand_tracking_push = reader.boolean();
                             if (reader.boolean())
                             {
                                 value.requirements.wrist_tracking_pull = read_wrist_config(reader);
                             }
                             return value;
                         });
}

HelloAck decode_hello_ack(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "HELLO_ACK",
                         [](Reader& reader)
                         {
                             HelloAck value;
                             value.accepted = reader.boolean();
                             value.reason = reader.str();
                             return value;
                         });
}

XrHandEXT decode_open_hand(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "OPEN_HAND", [](Reader& reader) { return from_wire_hand(reader.u8()); });
}

core::SchemaPusherConfig decode_open_schema(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "OPEN_SCHEMA",
                         [](Reader& reader)
                         {
                             core::SchemaPusherConfig value;
                             value.collection_id = reader.str();
                             value.max_flatbuffer_size = static_cast<size_t>(reader.u64());
                             value.tensor_identifier = reader.str();
                             value.localized_name = reader.str();
                             value.app_name = reader.str();
                             return value;
                         });
}

HandSample decode_hand_sample(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "HAND_SAMPLE",
                         [](Reader& reader)
                         {
                             HandSample value;
                             value.sample_time_ns = reader.i64();
                             for (XrHandJointLocationEXT& joint : value.joints)
                             {
                                 joint.locationFlags = reader.u64();
                                 joint.pose = reader.pose();
                                 joint.radius = reader.f32();
                             }
                             return value;
                         });
}

SchemaSample decode_schema_sample(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "SCHEMA_SAMPLE",
                         [](Reader& reader)
                         {
                             SchemaSample value;
                             value.sample_time_local_common_clock_ns = reader.i64();
                             value.sample_time_raw_device_clock_ns = reader.i64();
                             const uint32_t size = reader.u32();
                             value.bytes = reader.bytes(size);
                             return value;
                         });
}

WristSample decode_wrist_sample(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "WRIST_SAMPLE",
                         [](Reader& reader)
                         {
                             WristSample value;
                             value.sample_time_ns = reader.i64();
                             value.hand = reader.boolean() ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
                             value.sample.valid = reader.boolean();
                             value.sample.tracked = reader.boolean();
                             value.sample.pose = reader.pose();
                             return value;
                         });
}

std::string decode_error(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "ERROR", [](Reader& reader) { return reader.str(); });
}

int64_t decode_ping(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "PING", [](Reader& reader) { return reader.i64(); });
}

Pong decode_pong(std::span<const uint8_t> payload)
{
    return parse_payload(payload, "PONG",
                         [](Reader& reader)
                         {
                             Pong value;
                             value.ping_sent_ns = reader.i64();
                             value.ping_received_ns = reader.i64();
                             value.pong_sent_ns = reader.i64();
                             return value;
                         });
}

} // namespace remote_plugin_protocol
