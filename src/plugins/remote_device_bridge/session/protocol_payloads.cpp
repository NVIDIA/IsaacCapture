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

const char* to_string(FrameType type)
{
    switch (type)
    {
    case FrameType::Hello:
        return "HELLO";
    case FrameType::HelloAck:
        return "HELLO_ACK";
    case FrameType::OpenHand:
        return "OPEN_HAND";
    case FrameType::OpenSchema:
        return "OPEN_SCHEMA";
    case FrameType::OpenPull:
        return "OPEN_PULL";
    case FrameType::OpenWrist:
        return "OPEN_WRIST";
    case FrameType::Close:
        return "CLOSE";
    case FrameType::HandSample:
        return "HAND_SAMPLE";
    case FrameType::SchemaSample:
        return "SCHEMA_SAMPLE";
    case FrameType::WristSample:
        return "WRIST_SAMPLE";
    case FrameType::Heartbeat:
        return "HEARTBEAT";
    case FrameType::Error:
        return "ERROR";
    case FrameType::Ping:
        return "PING";
    case FrameType::Pong:
        return "PONG";
    }
    return "UNKNOWN";
}

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

// WRIST_SAMPLE keeps its original byte: 1 = left, 0 = right (OPEN_HAND uses 0 = left).
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

bool to_bool(uint8_t value)
{
    if (value > 1)
    {
        throw DecodeError("protocol: invalid boolean value " + std::to_string(value));
    }
    return value == 1;
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

} // namespace

std::vector<uint8_t> encode_hand_sample_frame(uint16_t channel_id,
                                              const XrHandJointLocationEXT* joints,
                                              int64_t sample_time_ns)
{
    return build_frame(FrameType::HandSample, channel_id, kHandSamplePayloadSize,
                       [&](Writer& writer)
                       {
                           writer.i64(sample_time_ns);
                           for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i)
                           {
                               writer.u64(joints[i].locationFlags);
                               writer.pose(joints[i].pose);
                               writer.f32(joints[i].radius);
                           }
                       });
}

std::vector<uint8_t> encode_schema_sample_frame(uint16_t channel_id,
                                                const uint8_t* bytes,
                                                size_t size,
                                                int64_t sample_time_local_common_clock_ns,
                                                int64_t sample_time_raw_device_clock_ns)
{
    if (size > kMaxPayloadSize)
    {
        throw std::invalid_argument("protocol: schema sample too large");
    }
    return build_frame(FrameType::SchemaSample, channel_id, 2 * sizeof(int64_t) + sizeof(uint32_t) + size,
                       [&](Writer& writer)
                       {
                           writer.i64(sample_time_local_common_clock_ns);
                           writer.i64(sample_time_raw_device_clock_ns);
                           writer.u32(static_cast<uint32_t>(size));
                           writer.bytes(bytes, size);
                       });
}

std::vector<uint8_t> encode_hello(const Hello& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.u16(value.proto_version);
    writer.str(value.app_name);
    writer.u8(value.requirements.schema_push ? 1 : 0);
    writer.u8(value.requirements.hand_tracking_push ? 1 : 0);
    writer.u8(value.requirements.wrist_tracking_pull ? 1 : 0);
    if (value.requirements.wrist_tracking_pull)
    {
        write_wrist_config(writer, *value.requirements.wrist_tracking_pull);
    }
    return out;
}

std::vector<uint8_t> encode_hello_ack(const HelloAck& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.u8(value.accepted ? 1 : 0);
    writer.str(value.reason);
    return out;
}

std::vector<uint8_t> encode_open_hand(XrHandEXT hand)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.u8(to_wire_hand(hand));
    return out;
}

std::vector<uint8_t> encode_open_schema(const core::SchemaPusherConfig& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.str(value.collection_id);
    writer.u64(static_cast<uint64_t>(value.max_flatbuffer_size));
    writer.str(value.tensor_identifier);
    writer.str(value.localized_name);
    writer.str(value.app_name);
    return out;
}

std::vector<uint8_t> encode_open_wrist(const core::WristTrackingSourceConfig& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    write_wrist_config(writer, value);
    return out;
}

std::vector<uint8_t> encode_wrist_sample(const WristSample& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.i64(value.sample_time_ns);
    writer.u8(to_wire_is_left(value.hand));
    writer.u8(value.sample.valid ? 1 : 0);
    writer.u8(value.sample.tracked ? 1 : 0);
    writer.pose(value.sample.pose);
    return out;
}

std::vector<uint8_t> encode_error(const std::string& message)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.str(message);
    return out;
}

std::vector<uint8_t> encode_ping(int64_t t1_ns)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.i64(t1_ns);
    return out;
}

std::vector<uint8_t> encode_pong(const Pong& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.i64(value.t1_ns);
    writer.i64(value.t2_ns);
    writer.i64(value.t3_ns);
    return out;
}

Hello decode_hello(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::Hello));
    Hello value;
    value.proto_version = reader.u16();
    if (value.proto_version < kHelloVersion)
    {
        throw DecodeError("protocol: HELLO version " + std::to_string(value.proto_version) + " is older than " +
                          std::to_string(kHelloVersion));
    }
    value.app_name = reader.str();
    value.requirements.schema_push = to_bool(reader.u8());
    value.requirements.hand_tracking_push = to_bool(reader.u8());
    if (to_bool(reader.u8()))
    {
        value.requirements.wrist_tracking_pull = read_wrist_config(reader);
    }
    reader.finish();
    return value;
}

HelloAck decode_hello_ack(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::HelloAck));
    HelloAck value;
    value.accepted = to_bool(reader.u8());
    value.reason = reader.str();
    reader.finish();
    return value;
}

XrHandEXT decode_open_hand(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::OpenHand));
    const XrHandEXT hand = from_wire_hand(reader.u8());
    reader.finish();
    return hand;
}

core::SchemaPusherConfig decode_open_schema(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::OpenSchema));
    core::SchemaPusherConfig value;
    value.collection_id = reader.str();
    value.max_flatbuffer_size = static_cast<size_t>(reader.u64());
    value.tensor_identifier = reader.str();
    value.localized_name = reader.str();
    value.app_name = reader.str();
    reader.finish();
    return value;
}

core::WristTrackingSourceConfig decode_open_wrist(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::OpenWrist));
    const core::WristTrackingSourceConfig value = read_wrist_config(reader);
    reader.finish();
    return value;
}

HandSample decode_hand_sample(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::HandSample));
    HandSample value;
    value.sample_time_ns = reader.i64();
    for (XrHandJointLocationEXT& joint : value.joints)
    {
        joint.locationFlags = reader.u64();
        joint.pose = reader.pose();
        joint.radius = reader.f32();
    }
    reader.finish();
    return value;
}

SchemaSample decode_schema_sample(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::SchemaSample));
    SchemaSample value;
    value.sample_time_local_common_clock_ns = reader.i64();
    value.sample_time_raw_device_clock_ns = reader.i64();
    const uint32_t size = reader.u32();
    value.bytes = reader.bytes(size);
    reader.finish();
    return value;
}

WristSample decode_wrist_sample(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::WristSample));
    WristSample value;
    value.sample_time_ns = reader.i64();
    value.hand = to_bool(reader.u8()) ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
    value.sample.valid = to_bool(reader.u8());
    value.sample.tracked = to_bool(reader.u8());
    value.sample.pose = reader.pose();
    reader.finish();
    return value;
}

std::string decode_error(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::Error));
    std::string message = reader.str();
    reader.finish();
    return message;
}

int64_t decode_ping(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::Ping));
    const int64_t t1_ns = reader.i64();
    reader.finish();
    return t1_ns;
}

Pong decode_pong(const std::vector<uint8_t>& payload)
{
    Reader reader(payload, to_string(FrameType::Pong));
    Pong value;
    value.t1_ns = reader.i64();
    value.t2_ns = reader.i64();
    value.t3_ns = reader.i64();
    reader.finish();
    return value;
}

} // namespace remote_plugin_protocol
