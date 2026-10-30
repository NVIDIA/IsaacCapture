// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "protocol_frame.hpp"

#include <openxr/openxr.h>
#include <pusherio/plugin_session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace remote_plugin_protocol
{

/*!
 * Payload builders and parsers. `str` = u32 length + bytes, `pose` = 7 x f32 in
 * XrPosef order (orientation x,y,z,w then position x,y,z).
 */

//! HELLO proto_version: the requirements carry the wrist source config, and the client answers PING with PONG.
//! decode_hello accepts only this version.
constexpr uint16_t kHelloVersion = 2;

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

//! On the wire the hand is one byte, 1 = left and 0 = right.
struct WristSample
{
    int64_t sample_time_ns = 0;
    XrHandEXT hand = XR_HAND_LEFT_EXT;
    core::WristTrackingSample sample;
};

//! Times are CLOCK_MONOTONIC ns of the side that took them.
struct Pong
{
    int64_t t1_ns = 0;
    int64_t t2_ns = 0;
    int64_t t3_ns = 0;
};

constexpr size_t kPoseSize = 7 * sizeof(float);
constexpr size_t kJointSize = sizeof(uint64_t) + kPoseSize + sizeof(float);
constexpr size_t kHandSamplePayloadSize = sizeof(int64_t) + XR_HAND_JOINT_COUNT_EXT * kJointSize;

static_assert(kJointSize == 40);
static_assert(kHandSamplePayloadSize == 1048);

std::vector<uint8_t> encode_hello(const Hello& value);
std::vector<uint8_t> encode_hello_ack(const HelloAck& value);
//! One byte: 0 = left, 1 = right.
std::vector<uint8_t> encode_open_hand(XrHandEXT hand);
std::vector<uint8_t> encode_open_schema(const core::SchemaPusherConfig& value);
std::vector<uint8_t> encode_wrist_sample(const WristSample& value);
std::vector<uint8_t> encode_error(const std::string& message);
std::vector<uint8_t> encode_ping(int64_t t1_ns);
std::vector<uint8_t> encode_pong(const Pong& value);
//! Sets t3 in an encoded PONG frame; call where the frame is sent, so t3 is when it left, not when it was queued.
void set_pong_t3(std::vector<uint8_t>& pong_frame, int64_t t3_ns);

// Sample frames are only built this way: the builders read the plugin's buffers directly so a push copies once.
std::vector<uint8_t> encode_hand_sample_frame(uint16_t channel_id,
                                              const XrHandJointLocationEXT* joints,
                                              int64_t sample_time_ns);
std::vector<uint8_t> encode_schema_sample_frame(uint16_t channel_id,
                                                const uint8_t* bytes,
                                                size_t size,
                                                int64_t sample_time_local_common_clock_ns,
                                                int64_t sample_time_raw_device_clock_ns);

// Parsers throw DecodeError, naming the frame, on truncated, over-long or out-of-range payloads.
Hello decode_hello(const std::vector<uint8_t>& payload);
HelloAck decode_hello_ack(const std::vector<uint8_t>& payload);
XrHandEXT decode_open_hand(const std::vector<uint8_t>& payload);
core::SchemaPusherConfig decode_open_schema(const std::vector<uint8_t>& payload);
HandSample decode_hand_sample(const std::vector<uint8_t>& payload);
SchemaSample decode_schema_sample(const std::vector<uint8_t>& payload);
WristSample decode_wrist_sample(const std::vector<uint8_t>& payload);
std::string decode_error(const std::vector<uint8_t>& payload);
int64_t decode_ping(const std::vector<uint8_t>& payload);
Pong decode_pong(const std::vector<uint8_t>& payload);

} // namespace remote_plugin_protocol
