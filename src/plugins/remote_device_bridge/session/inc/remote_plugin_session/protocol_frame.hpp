// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace remote_plugin_protocol
{

/*!
 * Remote plugin protocol v0: fixed little-endian frames over one connection per
 * plugin session. Frame = header + payload; payload layouts live in
 * protocol_payloads.hpp.
 */

constexpr uint32_t kMagic = 0x30535052u; // "RPS0"
constexpr uint8_t kProtocolVersion = 0;
//! magic u32, type u8, version u8, channel u16, seq u32, payload size u32, t_send i64.
constexpr size_t kHeaderSize =
    sizeof(uint32_t) + 2 * sizeof(uint8_t) + sizeof(uint16_t) + 2 * sizeof(uint32_t) + sizeof(int64_t);
constexpr uint16_t kSessionChannelId = 0;
constexpr uint32_t kMaxPayloadSize = 16u * 1024u * 1024u; // large tactile/sensor payloads; still bounded

//! Direction is c->s (plugin to bridge) or s->c. OPEN_PULL, CLOSE and HEARTBEAT have empty payloads.
enum class FrameType : uint8_t
{
    Hello = 1, // c->s
    HelloAck = 2, // s->c
    OpenHand = 3, // c->s
    OpenSchema = 4, // c->s
    OpenPull = 5, // c->s
    // 6 is unassigned: the wrist source config travels in HELLO.
    Close = 7, // c->s
    HandSample = 8, // c->s
    SchemaSample = 9, // c->s
    WristSample = 10, // s->c
    Heartbeat = 11, // both
    Error = 12, // s->c
    Ping = 13, // s->c
    Pong = 14, // c->s
};

struct FrameHeader
{
    FrameType type{}; // 0 is not a frame type
    uint16_t channel_id = kSessionChannelId;
    uint32_t seq = 0;
    uint32_t payload_size = 0;
    int64_t t_send_monotonic_ns = 0;
};

struct Frame
{
    FrameHeader header;
    std::vector<uint8_t> payload;
};

class DecodeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

//! Reads the header from the first kHeaderSize bytes of `in`. Throws DecodeError if `in` is shorter, or on bad
//! magic, unknown type, unsupported version or an oversized payload.
FrameHeader decode_header(std::span<const uint8_t> in);

//! Writes header + payload into one contiguous buffer. seq and t_send stay zero until stamp_header().
std::vector<uint8_t> encode_frame(FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload);
//! Patches seq and t_send into an encoded frame's header in place.
void stamp_header(std::vector<uint8_t>& frame, uint32_t seq, int64_t t_send_monotonic_ns);
//! Splits one complete message into header and payload. The payload keeps `message`'s buffer (its bytes are
//! shifted down by the header size). Throws DecodeError.
Frame split_frame(std::vector<uint8_t> message);

} // namespace remote_plugin_protocol
