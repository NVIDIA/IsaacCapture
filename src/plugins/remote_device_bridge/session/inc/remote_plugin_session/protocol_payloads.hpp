// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "protocol_frame.hpp"

#include <openxr/openxr.h>
#include <pusherio/plugin_session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace remote_plugin_protocol
{

/*!
 * Payload builders and parsers. `str` = u32 length + bytes, `pose` = 7 x f32 in
 * XrPosef order (orientation x,y,z,w then position x,y,z).
 */

struct Hello
{
    std::string app_name;
    core::PluginSessionRequirements requirements;
};

struct HelloAck
{
    bool accepted = false;
    std::string reason;
};

struct HandSample
{
    int64_t sample_time_local_common_clock_ns = 0;
    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
};

struct SchemaSample
{
    int64_t sample_time_local_common_clock_ns = 0;
    int64_t sample_time_raw_device_clock_ns = 0;
    std::vector<uint8_t> bytes;
};

//! On the wire the hand is one byte, 0 = left and 1 = right, as in OPEN_HAND.
struct WristSample
{
    int64_t sample_time_local_common_clock_ns = 0;
    XrHandEXT hand = XR_HAND_LEFT_EXT;
    core::WristTrackingSample sample;
};

//! Two-way clock exchange; times are CLOCK_MONOTONIC ns of the side that took them. The PING carries no payload:
//! its send time is its header t_send. The PONG's own send time is the PONG header t_send.
struct Pong
{
    int64_t ping_sent_ns = 0; //!< bridge clock: the PING header t_send, echoed
    int64_t ping_received_ns = 0; //!< client clock
};

constexpr size_t kPoseSize = 7 * sizeof(float);
constexpr size_t kJointSize = sizeof(uint64_t) + kPoseSize + sizeof(float);
constexpr size_t kHandSamplePayloadSize = sizeof(int64_t) + XR_HAND_JOINT_COUNT_EXT * kJointSize;
//! SCHEMA_SAMPLE payload before the schema bytes: two sample times (i64) and the byte count (u32).
constexpr size_t kSchemaSampleFixedSize = 2 * sizeof(int64_t) + sizeof(uint32_t);

static_assert(kJointSize == 40);
static_assert(kHandSamplePayloadSize == 1048);
static_assert(kSchemaSampleFixedSize == 20);

//! One hand's joints, as IHandTrackingPushChannel::push receives them.
using HandJoints = std::span<const XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT>;

std::vector<uint8_t> encode_hello(const Hello& value);
std::vector<uint8_t> encode_hello_ack(const HelloAck& value);
//! One byte: 0 = left, 1 = right.
std::vector<uint8_t> encode_open_hand(XrHandEXT hand);
std::vector<uint8_t> encode_open_schema(const core::SchemaPusherConfig& value);
std::vector<uint8_t> encode_wrist_sample(const WristSample& value);
std::vector<uint8_t> encode_error(const std::string& message);
std::vector<uint8_t> encode_pong(const Pong& value);

// Sample frames are only built this way: the builders read the plugin's buffers directly so a push copies once.
std::vector<uint8_t> encode_hand_sample_frame(uint16_t channel_id,
                                              HandJoints joints,
                                              int64_t sample_time_local_common_clock_ns);
std::vector<uint8_t> encode_schema_sample_frame(uint16_t channel_id,
                                                std::span<const uint8_t> bytes,
                                                int64_t sample_time_local_common_clock_ns,
                                                int64_t sample_time_raw_device_clock_ns);

// Parsers throw DecodeError, naming the frame, on truncated, over-long or out-of-range payloads.
Hello decode_hello(std::span<const uint8_t> payload);
HelloAck decode_hello_ack(std::span<const uint8_t> payload);
XrHandEXT decode_open_hand(std::span<const uint8_t> payload);
core::SchemaPusherConfig decode_open_schema(std::span<const uint8_t> payload);
HandSample decode_hand_sample(std::span<const uint8_t> payload);
SchemaSample decode_schema_sample(std::span<const uint8_t> payload);
WristSample decode_wrist_sample(std::span<const uint8_t> payload);
std::string decode_error(std::span<const uint8_t> payload);
Pong decode_pong(std::span<const uint8_t> payload);

} // namespace remote_plugin_protocol
