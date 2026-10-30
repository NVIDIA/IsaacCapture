// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openxr/openxr.h>
#include <pusherio/plugin_session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace remote_plugin_session
{
namespace wire
{

/*!
 * Wire protocol v0: fixed little-endian frames over one TCP connection per
 * plugin session. Frame = 24-byte header + payload. `str` = u32 length + bytes,
 * `pose` = 7 x f32 in XrPosef order (orientation x,y,z,w then position x,y,z).
 * This file is the single serialization point; swapping the hand payload for
 * a FlatBuffer later only changes encode/decode of HandSample here.
 */

constexpr uint32_t kMagic = 0x30535052u; // "RPS0"
constexpr uint8_t kProtocolVersion = 0;
//! HELLO proto_version: 2 = requirements carry the wrist source config (not a flag) and the client answers
//! PING with PONG. decode_hello rejects older versions, whose payload layout differs.
constexpr uint16_t kHelloVersion = 2;
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
    WristSample = 10, // s->c: sample_time, is_left, valid, tracked, pose
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

struct Hello
{
    uint16_t proto_version = kHelloVersion;
    std::string app_name;
    core::PluginSessionRequirements requirements;
};

struct HelloAck
{
    bool accepted = false;
    std::string reason;
};

struct OpenHand
{
    XrHandEXT hand = XR_HAND_LEFT_EXT;
};

struct HandSample
{
    int64_t sample_time_ns = 0;
    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
};

struct SchemaSample
{
    int64_t sample_time_local_common_clock_ns = 0;
    int64_t sample_time_raw_device_clock_ns = 0;
    std::vector<uint8_t> bytes;
};

struct WristSample
{
    int64_t sample_time_ns = 0;
    bool is_left = true;
    core::WristTrackingSample sample;
};

struct Error
{
    std::string message;
};

//! Times are CLOCK_MONOTONIC ns of the side that took them.
struct Ping
{
    int64_t t1_ns = 0;
};

struct Pong
{
    int64_t t1_ns = 0;
    int64_t t2_ns = 0;
    int64_t t3_ns = 0;
};

constexpr size_t kPoseSize = 7 * sizeof(float);
constexpr size_t kJointSize = sizeof(uint64_t) + kPoseSize + sizeof(float);
constexpr size_t kHandSamplePayloadSize = sizeof(int64_t) + XR_HAND_JOINT_COUNT_EXT * kJointSize;
constexpr size_t kWristSamplePayloadSize = sizeof(int64_t) + 3 * sizeof(uint8_t) + kPoseSize;
constexpr size_t kWristConfigPayloadSize = sizeof(uint8_t) + 2 * kPoseSize;

static_assert(kHeaderSize == 4 + 1 + 1 + 2 + 4 + 4 + 8);
static_assert(kJointSize == 40);
static_assert(kHandSamplePayloadSize == 1048);
static_assert(sizeof(XrSpaceLocationFlags) == sizeof(uint64_t));

const char* to_string(FrameType type);

//! `out` must hold kHeaderSize bytes.
void encode_header(const FrameHeader& header, uint8_t* out);
//! Throws DecodeError on bad magic, unsupported version or an oversized payload.
FrameHeader decode_header(const uint8_t* in);

std::vector<uint8_t> encode(const Hello& value);
std::vector<uint8_t> encode(const HelloAck& value);
std::vector<uint8_t> encode(const OpenHand& value);
std::vector<uint8_t> encode(const core::SchemaPusherConfig& value);
std::vector<uint8_t> encode(const core::WristTrackingSourceConfig& value);
std::vector<uint8_t> encode(const WristSample& value);
std::vector<uint8_t> encode(const Error& value);
std::vector<uint8_t> encode(const Ping& value);
std::vector<uint8_t> encode(const Pong& value);

// Frame builders write header + payload into one contiguous buffer. seq and t_send stay zero until
// stamp_header(). Samples are only encoded this way: the builders read the plugin's buffers directly so a
// push copies once.
std::vector<uint8_t> encode_frame(FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload);
std::vector<uint8_t> encode_hand_sample_frame(uint16_t channel_id,
                                              const XrHandJointLocationEXT* joints,
                                              int64_t sample_time_ns);
std::vector<uint8_t> encode_schema_sample_frame(uint16_t channel_id,
                                                const uint8_t* bytes,
                                                size_t size,
                                                int64_t sample_time_local_common_clock_ns,
                                                int64_t sample_time_raw_device_clock_ns);
//! Patches seq and t_send into an encoded frame's header in place; on a PONG also sets t3 = t_send, so t3 is
//! the time the frame actually left, not when it was queued.
void stamp_header(uint8_t* frame, uint32_t seq, int64_t t_send_monotonic_ns);
//! Splits one complete message into header and payload (payload moved, not copied). Throws DecodeError.
Frame split_frame(std::vector<uint8_t> message);

// Decoders throw DecodeError on truncated or over-long payloads.
Hello decode_hello(const std::vector<uint8_t>& payload);
HelloAck decode_hello_ack(const std::vector<uint8_t>& payload);
OpenHand decode_open_hand(const std::vector<uint8_t>& payload);
core::SchemaPusherConfig decode_open_schema(const std::vector<uint8_t>& payload);
core::WristTrackingSourceConfig decode_open_wrist(const std::vector<uint8_t>& payload);
HandSample decode_hand_sample(const std::vector<uint8_t>& payload);
SchemaSample decode_schema_sample(const std::vector<uint8_t>& payload);
WristSample decode_wrist_sample(const std::vector<uint8_t>& payload);
Error decode_error(const std::vector<uint8_t>& payload);
Ping decode_ping(const std::vector<uint8_t>& payload);
Pong decode_pong(const std::vector<uint8_t>& payload);

} // namespace wire
} // namespace remote_plugin_session
