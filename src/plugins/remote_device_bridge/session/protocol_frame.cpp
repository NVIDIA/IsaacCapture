// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/protocol_frame.hpp"

#include "protocol_io.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace remote_plugin_protocol
{

using detail::build_frame;
using detail::Reader;
using detail::Writer;

namespace
{

constexpr uint8_t kUnassignedFrameType = 6;
constexpr size_t kTypeOffset = 4;
constexpr size_t kVersionOffset = 5;
constexpr size_t kChannelOffset = 6;
constexpr size_t kSeqOffset = 8;
constexpr size_t kPayloadSizeOffset = 12;
constexpr size_t kSendTimeOffset = 16;

void store_le(uint8_t* out, uint64_t value, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        out[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

} // namespace

namespace detail
{

void encode_header(const FrameHeader& header, uint8_t* out)
{
    store_le(out, kMagic, sizeof(uint32_t));
    out[kTypeOffset] = static_cast<uint8_t>(header.type);
    out[kVersionOffset] = header.version;
    store_le(out + kChannelOffset, header.channel_id, sizeof(uint16_t));
    store_le(out + kSeqOffset, header.seq, sizeof(uint32_t));
    store_le(out + kPayloadSizeOffset, header.payload_size, sizeof(uint32_t));
    store_le(out + kSendTimeOffset, static_cast<uint64_t>(header.t_send_monotonic_ns), sizeof(int64_t));
}

} // namespace detail

void stamp_header(std::vector<uint8_t>& frame, uint32_t seq, int64_t t_send_monotonic_ns)
{
    if (frame.size() < kHeaderSize)
    {
        throw std::invalid_argument("protocol: frame shorter than a header");
    }
    store_le(frame.data() + kSeqOffset, seq, sizeof(uint32_t));
    store_le(frame.data() + kSendTimeOffset, static_cast<uint64_t>(t_send_monotonic_ns), sizeof(int64_t));
}

Frame split_frame(std::vector<uint8_t> message)
{
    if (message.size() < kHeaderSize)
    {
        throw DecodeError("protocol: message shorter than a header");
    }
    Frame frame;
    frame.header = decode_header(message.data());
    if (message.size() != kHeaderSize + frame.header.payload_size)
    {
        throw DecodeError("protocol: message length does not match payload_size");
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

FrameHeader decode_header(const uint8_t* in)
{
    Reader reader(in, kHeaderSize, "header");
    if (reader.u32() != kMagic)
    {
        throw DecodeError("protocol: bad frame magic");
    }
    FrameHeader header;
    const uint8_t type = reader.u8();
    if (type < static_cast<uint8_t>(FrameType::Hello) || type > static_cast<uint8_t>(kLastFrameType) ||
        type == kUnassignedFrameType)
    {
        throw DecodeError("protocol: unknown frame type " + std::to_string(type));
    }
    header.type = static_cast<FrameType>(type);
    header.version = reader.u8();
    if (header.version != kProtocolVersion)
    {
        throw DecodeError("protocol: unsupported protocol version " + std::to_string(header.version));
    }
    header.channel_id = reader.u16();
    header.seq = reader.u32();
    header.payload_size = reader.u32();
    if (header.payload_size > kMaxPayloadSize)
    {
        throw DecodeError("protocol: payload size " + std::to_string(header.payload_size) + " exceeds limit");
    }
    header.t_send_monotonic_ns = reader.i64();
    return header;
}

} // namespace remote_plugin_protocol
