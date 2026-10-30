// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace remote_plugin_protocol
{

/*!
 * Remote plugin protocol v0: fixed little-endian frames over one connection per
 * plugin session. Frame = 24-byte header + payload. Payload layouts live in
 * protocol_payloads.hpp.
 */

constexpr uint32_t kMagic = 0x30535052u; // "RPS0"
constexpr uint8_t kProtocolVersion = 0;
constexpr size_t kHeaderSize = 24;
constexpr uint16_t kSessionChannelId = 0;
constexpr uint32_t kMaxPayloadSize = 16u * 1024u * 1024u; // large tactile/sensor payloads; still bounded

enum class FrameType : uint8_t
{
    Hello = 1, // c->s: proto_version, app_name, requirements
    HelloAck = 2, // s->c: accepted, reason
    OpenHand = 3, // c->s: hand (0 left, 1 right)
    OpenSchema = 4, // c->s: SchemaPusherConfig
    OpenPull = 5, // c->s: empty
    OpenWrist = 6, // c->s: WristTrackingSourceConfig on the pull channel id
    Close = 7, // c->s: empty
    HandSample = 8, // c->s: sample_time + 26 joints
    SchemaSample = 9, // c->s: two timestamps + bytes
    WristSample = 10, // s->c: sample_time, hand, valid, tracked, pose
    Heartbeat = 11, // both: empty
    Error = 12, // s->c: message; on channel 0 the connection closes afterwards, on a channel id only that
                // channel is refused
    Ping = 13, // s->c: t1 (bridge send time)
    Pong = 14, // c->s: t1 echoed, t2 (client receive time), t3 (client send time)
};

struct FrameHeader
{
    FrameType type = FrameType::Heartbeat;
    uint8_t version = kProtocolVersion;
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

static_assert(kHeaderSize == 4 + 1 + 1 + 2 + 4 + 4 + 8);

//! `out` must hold kHeaderSize bytes.
void encode_header(const FrameHeader& header, uint8_t* out);
//! Throws DecodeError on bad magic, unknown type, unsupported version or an oversized payload.
FrameHeader decode_header(const uint8_t* in);

//! Writes header + payload into one contiguous buffer. seq and t_send stay zero until stamp_header().
std::vector<uint8_t> encode_frame(FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload);
//! Patches seq and t_send into an encoded frame's header in place; on a PONG also sets t3 = t_send, so t3 is
//! the time the frame actually left, not when it was queued.
void stamp_header(uint8_t* frame, uint32_t seq, int64_t t_send_monotonic_ns);
//! Splits one complete message into header and payload (payload moved, not copied). Throws DecodeError.
Frame split_frame(std::vector<uint8_t> message);

} // namespace remote_plugin_protocol
