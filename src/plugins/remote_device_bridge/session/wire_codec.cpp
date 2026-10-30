// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/wire_codec.hpp"

#include <cstring>
#include <limits>

namespace remote_plugin_session
{
namespace wire
{

namespace
{

//! Appends little-endian fields to a caller-owned buffer.
class Writer
{
public:
    explicit Writer(std::vector<uint8_t>& out) : m_out(out)
    {
    }

    void u8(uint8_t value)
    {
        m_out.push_back(value);
    }

    void u16(uint16_t value)
    {
        for (int i = 0; i < 2; ++i)
        {
            m_out.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    }

    void u32(uint32_t value)
    {
        for (int i = 0; i < 4; ++i)
        {
            m_out.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    }

    void u64(uint64_t value)
    {
        for (int i = 0; i < 8; ++i)
        {
            m_out.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    }

    void i64(int64_t value)
    {
        u64(static_cast<uint64_t>(value));
    }

    void f32(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u32(bits);
    }

    void str(const std::string& value)
    {
        if (value.size() > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("wire: string too long");
        }
        u32(static_cast<uint32_t>(value.size()));
        m_out.insert(m_out.end(), value.begin(), value.end());
    }

    void bytes(const uint8_t* data, size_t size)
    {
        m_out.insert(m_out.end(), data, data + size);
    }

    void pose(const XrPosef& value)
    {
        f32(value.orientation.x);
        f32(value.orientation.y);
        f32(value.orientation.z);
        f32(value.orientation.w);
        f32(value.position.x);
        f32(value.position.y);
        f32(value.position.z);
    }

private:
    std::vector<uint8_t>& m_out;
};

class Reader
{
public:
    Reader(const uint8_t* data, size_t size) : m_data(data), m_size(size)
    {
    }

    explicit Reader(const std::vector<uint8_t>& payload) : Reader(payload.data(), payload.size())
    {
    }

    uint8_t u8()
    {
        need(1);
        return m_data[m_pos++];
    }

    uint16_t u16()
    {
        need(2);
        uint16_t value = 0;
        for (int i = 0; i < 2; ++i)
        {
            value = static_cast<uint16_t>(value | (static_cast<uint16_t>(m_data[m_pos++]) << (8 * i)));
        }
        return value;
    }

    uint32_t u32()
    {
        need(4);
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i)
        {
            value |= static_cast<uint32_t>(m_data[m_pos++]) << (8 * i);
        }
        return value;
    }

    uint64_t u64()
    {
        need(8);
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i)
        {
            value |= static_cast<uint64_t>(m_data[m_pos++]) << (8 * i);
        }
        return value;
    }

    int64_t i64()
    {
        return static_cast<int64_t>(u64());
    }

    float f32()
    {
        const uint32_t bits = u32();
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::string str()
    {
        const uint32_t size = u32();
        need(size);
        std::string value(reinterpret_cast<const char*>(m_data + m_pos), size);
        m_pos += size;
        return value;
    }

    std::vector<uint8_t> bytes(size_t size)
    {
        need(size);
        std::vector<uint8_t> value(m_data + m_pos, m_data + m_pos + size);
        m_pos += size;
        return value;
    }

    XrPosef pose()
    {
        XrPosef value;
        value.orientation.x = f32();
        value.orientation.y = f32();
        value.orientation.z = f32();
        value.orientation.w = f32();
        value.position.x = f32();
        value.position.y = f32();
        value.position.z = f32();
        return value;
    }

    void finish() const
    {
        if (m_pos != m_size)
        {
            throw DecodeError("wire: " + std::to_string(m_size - m_pos) + " trailing bytes");
        }
    }

private:
    void need(size_t count) const
    {
        if (m_size - m_pos < count)
        {
            throw DecodeError("wire: truncated payload");
        }
    }

    const uint8_t* m_data;
    size_t m_size;
    size_t m_pos = 0;
};

uint8_t to_wire_hand(XrHandEXT hand)
{
    switch (hand)
    {
    case XR_HAND_LEFT_EXT:
        return 0;
    case XR_HAND_RIGHT_EXT:
        return 1;
    default:
        throw std::invalid_argument("wire: unsupported XrHandEXT value");
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
        throw DecodeError("wire: invalid hand value " + std::to_string(hand));
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
    throw std::invalid_argument("wire: unsupported wrist tracking source mode");
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
        throw DecodeError("wire: invalid wrist tracking source mode " + std::to_string(mode));
    }
}

//! Reserves the header, writes the payload behind it, then fills the header in with the resulting size.
template <typename WritePayload>
std::vector<uint8_t> build_frame(FrameType type, uint16_t channel_id, size_t payload_hint, WritePayload write_payload)
{
    std::vector<uint8_t> out;
    out.reserve(kHeaderSize + payload_hint);
    out.resize(kHeaderSize);
    Writer writer(out);
    write_payload(writer);
    if (out.size() - kHeaderSize > kMaxPayloadSize)
    {
        throw std::invalid_argument("wire: payload too large");
    }
    FrameHeader header;
    header.type = type;
    header.channel_id = channel_id;
    header.payload_size = static_cast<uint32_t>(out.size() - kHeaderSize);
    encode_header(header, out.data());
    return out;
}

bool to_bool(uint8_t value)
{
    if (value > 1)
    {
        throw DecodeError("wire: invalid boolean value " + std::to_string(value));
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

void encode_header(const FrameHeader& header, uint8_t* out)
{
    std::vector<uint8_t> bytes;
    Writer writer(bytes);
    writer.u32(kMagic);
    writer.u8(static_cast<uint8_t>(header.type));
    writer.u8(header.version);
    writer.u16(header.channel_id);
    writer.u32(header.seq);
    writer.u32(header.payload_size);
    writer.i64(header.t_send_monotonic_ns);
    std::memcpy(out, bytes.data(), kHeaderSize);
}

void stamp_header(uint8_t* frame, uint32_t seq, int64_t t_send_monotonic_ns)
{
    // Byte offsets follow encode_header: seq at 8, t_send at 16.
    for (int i = 0; i < 4; ++i)
    {
        frame[8 + i] = static_cast<uint8_t>(seq >> (8 * i));
    }
    const uint64_t t_send = static_cast<uint64_t>(t_send_monotonic_ns);
    for (int i = 0; i < 8; ++i)
    {
        frame[16 + i] = static_cast<uint8_t>(t_send >> (8 * i));
    }
    // Pong payload: t1, t2, t3 (i64 each); t3 at payload offset 16.
    if (frame[4] == static_cast<uint8_t>(FrameType::Pong) && decode_header(frame).payload_size == 3 * sizeof(int64_t))
    {
        for (int i = 0; i < 8; ++i)
        {
            frame[kHeaderSize + 16 + i] = static_cast<uint8_t>(t_send >> (8 * i));
        }
    }
}

Frame split_frame(std::vector<uint8_t> message)
{
    if (message.size() < kHeaderSize)
    {
        throw DecodeError("wire: message shorter than a header");
    }
    Frame frame;
    frame.header = decode_header(message.data());
    if (message.size() != kHeaderSize + frame.header.payload_size)
    {
        throw DecodeError("wire: message length does not match payload_size");
    }
    message.erase(message.begin(), message.begin() + static_cast<std::ptrdiff_t>(kHeaderSize));
    frame.payload = std::move(message);
    return frame;
}

std::vector<uint8_t> encode_frame(FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload)
{
    return build_frame(
        type, channel_id, payload.size(), [&payload](Writer& writer) { writer.bytes(payload.data(), payload.size()); });
}

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
        throw std::invalid_argument("wire: schema sample too large");
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

FrameHeader decode_header(const uint8_t* in)
{
    Reader reader(in, kHeaderSize);
    if (reader.u32() != kMagic)
    {
        throw DecodeError("wire: bad frame magic");
    }
    FrameHeader header;
    const uint8_t type = reader.u8();
    if (type < static_cast<uint8_t>(FrameType::Hello) || type > static_cast<uint8_t>(FrameType::Pong))
    {
        throw DecodeError("wire: unknown frame type " + std::to_string(type));
    }
    header.type = static_cast<FrameType>(type);
    header.version = reader.u8();
    if (header.version != kProtocolVersion)
    {
        throw DecodeError("wire: unsupported protocol version " + std::to_string(header.version));
    }
    header.channel_id = reader.u16();
    header.seq = reader.u32();
    header.payload_size = reader.u32();
    if (header.payload_size > kMaxPayloadSize)
    {
        throw DecodeError("wire: payload size " + std::to_string(header.payload_size) + " exceeds limit");
    }
    header.t_send_monotonic_ns = reader.i64();
    reader.finish();
    return header;
}

std::vector<uint8_t> encode(const Hello& value)
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

std::vector<uint8_t> encode(const HelloAck& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.u8(value.accepted ? 1 : 0);
    writer.str(value.reason);
    return out;
}

std::vector<uint8_t> encode(const OpenHand& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.u8(to_wire_hand(value.hand));
    return out;
}

std::vector<uint8_t> encode(const core::SchemaPusherConfig& value)
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

std::vector<uint8_t> encode(const core::WristTrackingSourceConfig& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    write_wrist_config(writer, value);
    return out;
}

std::vector<uint8_t> encode(const WristSample& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.i64(value.sample_time_ns);
    writer.u8(value.is_left ? 1 : 0);
    writer.u8(value.sample.valid ? 1 : 0);
    writer.u8(value.sample.tracked ? 1 : 0);
    writer.pose(value.sample.pose);
    return out;
}

std::vector<uint8_t> encode(const Error& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.str(value.message);
    return out;
}

std::vector<uint8_t> encode(const Ping& value)
{
    std::vector<uint8_t> out;
    Writer writer(out);
    writer.i64(value.t1_ns);
    return out;
}

std::vector<uint8_t> encode(const Pong& value)
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
    Reader reader(payload);
    Hello value;
    value.proto_version = reader.u16();
    if (value.proto_version < kHelloVersion)
    {
        throw DecodeError("wire: HELLO version " + std::to_string(value.proto_version) + " is older than " +
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
    Reader reader(payload);
    HelloAck value;
    value.accepted = to_bool(reader.u8());
    value.reason = reader.str();
    reader.finish();
    return value;
}

OpenHand decode_open_hand(const std::vector<uint8_t>& payload)
{
    Reader reader(payload);
    OpenHand value;
    value.hand = from_wire_hand(reader.u8());
    reader.finish();
    return value;
}

core::SchemaPusherConfig decode_open_schema(const std::vector<uint8_t>& payload)
{
    Reader reader(payload);
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
    Reader reader(payload);
    const core::WristTrackingSourceConfig value = read_wrist_config(reader);
    reader.finish();
    return value;
}

HandSample decode_hand_sample(const std::vector<uint8_t>& payload)
{
    Reader reader(payload);
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
    Reader reader(payload);
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
    Reader reader(payload);
    WristSample value;
    value.sample_time_ns = reader.i64();
    value.is_left = to_bool(reader.u8());
    value.sample.valid = to_bool(reader.u8());
    value.sample.tracked = to_bool(reader.u8());
    value.sample.pose = reader.pose();
    reader.finish();
    return value;
}

Error decode_error(const std::vector<uint8_t>& payload)
{
    Reader reader(payload);
    Error value;
    value.message = reader.str();
    reader.finish();
    return value;
}

Ping decode_ping(const std::vector<uint8_t>& payload)
{
    Reader reader(payload);
    Ping value;
    value.t1_ns = reader.i64();
    reader.finish();
    return value;
}

Pong decode_pong(const std::vector<uint8_t>& payload)
{
    Reader reader(payload);
    Pong value;
    value.t1_ns = reader.i64();
    value.t2_ns = reader.i64();
    value.t3_ns = reader.i64();
    reader.finish();
    return value;
}

} // namespace wire
} // namespace remote_plugin_session
