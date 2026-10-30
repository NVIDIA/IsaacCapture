// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/protocol_frame.hpp"

#include "protocol_io.hpp"

#include <cstring>
#include <string>
#include <utility>

namespace remote_plugin_protocol
{

using detail::build_frame;
using detail::Reader;
using detail::Writer;

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
    if (type < static_cast<uint8_t>(FrameType::Hello) || type > static_cast<uint8_t>(FrameType::Pong))
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
    reader.finish();
    return header;
}

} // namespace remote_plugin_protocol
