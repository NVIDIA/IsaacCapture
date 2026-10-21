// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <remote_plugin_session/wire_codec.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace remote_plugin_session;

namespace
{

bool same_pose(const XrPosef& a, const XrPosef& b)
{
    return std::memcmp(&a, &b, sizeof(XrPosef)) == 0;
}

XrPosef make_pose(float seed)
{
    return { { seed, seed + 0.25f, -seed, 0.5f }, { 10.0f * seed, -20.0f * seed, 30.0f * seed } };
}

void put_u32(std::vector<uint8_t>& out, uint32_t v)
{
    for (int i = 0; i < 4; ++i)
    {
        out.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
}

void put_u64(std::vector<uint8_t>& out, uint64_t v)
{
    put_u32(out, static_cast<uint32_t>(v));
    put_u32(out, static_cast<uint32_t>(v >> 32));
}

void put_f32(std::vector<uint8_t>& out, float v)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    put_u32(out, bits);
}

//! Reference payload encoder (the pre-frame-builder layout) kept as the golden for HAND_SAMPLE.
std::vector<uint8_t> reference_hand_payload(const wire::HandSample& sample)
{
    std::vector<uint8_t> out;
    put_u64(out, static_cast<uint64_t>(sample.sample_time_ns));
    for (const XrHandJointLocationEXT& joint : sample.joints)
    {
        put_u64(out, joint.locationFlags);
        for (const float f :
             { joint.pose.orientation.x, joint.pose.orientation.y, joint.pose.orientation.z, joint.pose.orientation.w,
               joint.pose.position.x, joint.pose.position.y, joint.pose.position.z, joint.radius })
        {
            put_f32(out, f);
        }
    }
    return out;
}

std::vector<uint8_t> hand_payload(const wire::HandSample& sample)
{
    return wire::split_frame(wire::encode_hand_sample_frame(1, sample.joints.data(), sample.sample_time_ns)).payload;
}

std::vector<uint8_t> schema_payload(const wire::SchemaSample& sample)
{
    return wire::split_frame(wire::encode_schema_sample_frame(1, sample.bytes.data(), sample.bytes.size(),
                                                              sample.sample_time_local_common_clock_ns,
                                                              sample.sample_time_raw_device_clock_ns))
        .payload;
}

} // namespace

TEST_CASE("Frame header round-trips and validates", "[remote_device_bridge][unit]")
{
    wire::FrameHeader header;
    header.type = wire::FrameType::HandSample;
    header.channel_id = 0xBEEF;
    header.seq = 0x01020304;
    header.payload_size = static_cast<uint32_t>(wire::kHandSamplePayloadSize);
    header.t_send_monotonic_ns = -1234567890123LL;

    uint8_t bytes[wire::kHeaderSize];
    wire::encode_header(header, bytes);
    REQUIRE(bytes[0] == 0x52); // little-endian "RPS0"
    REQUIRE(bytes[1] == 0x50);
    REQUIRE(bytes[2] == 0x53);
    REQUIRE(bytes[3] == 0x30);

    const wire::FrameHeader decoded = wire::decode_header(bytes);
    REQUIRE(decoded.type == wire::FrameType::HandSample);
    REQUIRE(decoded.version == wire::kProtocolVersion);
    REQUIRE(decoded.channel_id == 0xBEEF);
    REQUIRE(decoded.seq == 0x01020304);
    REQUIRE(decoded.payload_size == wire::kHandSamplePayloadSize);
    REQUIRE(decoded.t_send_monotonic_ns == -1234567890123LL);

    SECTION("bad magic is rejected")
    {
        bytes[0] ^= 0xFF;
        REQUIRE_THROWS_AS(wire::decode_header(bytes), wire::DecodeError);
    }
    SECTION("unknown type is rejected")
    {
        bytes[4] = 200;
        REQUIRE_THROWS_AS(wire::decode_header(bytes), wire::DecodeError);
    }
    SECTION("other protocol versions are rejected")
    {
        bytes[5] = 1;
        REQUIRE_THROWS_AS(wire::decode_header(bytes), wire::DecodeError);
    }
    SECTION("oversized payloads are rejected")
    {
        std::memset(bytes + 12, 0xFF, 4);
        REQUIRE_THROWS_AS(wire::decode_header(bytes), wire::DecodeError);
    }
}

TEST_CASE("Wire contract v1 is 12 message types plus PING and PONG in a 24-byte header", "[remote_device_bridge][unit]")
{
    const std::vector<std::pair<wire::FrameType, std::string>> contract = {
        { wire::FrameType::Hello, "HELLO" },
        { wire::FrameType::HelloAck, "HELLO_ACK" },
        { wire::FrameType::OpenHand, "OPEN_HAND" },
        { wire::FrameType::OpenSchema, "OPEN_SCHEMA" },
        { wire::FrameType::OpenPull, "OPEN_PULL" },
        { wire::FrameType::OpenWrist, "OPEN_WRIST" },
        { wire::FrameType::Close, "CLOSE" },
        { wire::FrameType::HandSample, "HAND_SAMPLE" },
        { wire::FrameType::SchemaSample, "SCHEMA_SAMPLE" },
        { wire::FrameType::WristSample, "WRIST_SAMPLE" },
        { wire::FrameType::Heartbeat, "HEARTBEAT" },
        { wire::FrameType::Error, "ERROR" },
        { wire::FrameType::Ping, "PING" }, // bridge -> sender, t1
        { wire::FrameType::Pong, "PONG" }, // sender -> bridge, t1 t2 t3
    };
    for (size_t i = 0; i < contract.size(); ++i)
    {
        REQUIRE(static_cast<uint8_t>(contract[i].first) == i + 1);
        REQUIRE(std::string(wire::to_string(contract[i].first)) == contract[i].second);
    }
    REQUIRE(wire::kHeaderSize == 24);
    REQUIRE(wire::kProtocolVersion == 0);
    REQUIRE(wire::kHelloVersion == 1);

    // magic u32 @0, type u8 @4, version u8 @5, channel u16 @6, seq u32 @8, payload size u32 @12, t_send i64 @16.
    wire::FrameHeader header;
    header.type = wire::FrameType::Pong;
    header.channel_id = 0x0A0B;
    header.seq = 0x0C0D0E0F;
    header.payload_size = 24;
    header.t_send_monotonic_ns = 0x1112131415161718LL;
    uint8_t bytes[wire::kHeaderSize];
    wire::encode_header(header, bytes);
    const std::vector<uint8_t> expected = { 0x52, 0x50, 0x53, 0x30, 14,   0,    0x0B, 0x0A, 0x0F, 0x0E, 0x0D, 0x0C,
                                            24,   0,    0,    0,    0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11 };
    REQUIRE(std::vector<uint8_t>(bytes, bytes + wire::kHeaderSize) == expected);

    for (const uint8_t type : { 0, 15 })
    {
        bytes[4] = type;
        REQUIRE_THROWS_AS(wire::decode_header(bytes), wire::DecodeError);
    }
}

TEST_CASE("HELLO and HELLO_ACK round-trip", "[remote_device_bridge][unit]")
{
    wire::Hello hello;
    hello.app_name = "ManusHandPlugin";
    hello.requirements = { .schema_push = true, .hand_tracking_push = false, .wrist_tracking_pull = true };
    const wire::Hello decoded = wire::decode_hello(wire::encode(hello));
    REQUIRE(decoded.proto_version == wire::kHelloVersion);
    REQUIRE(decoded.app_name == "ManusHandPlugin");
    REQUIRE(decoded.requirements.schema_push);
    REQUIRE_FALSE(decoded.requirements.hand_tracking_push);
    REQUIRE(decoded.requirements.wrist_tracking_pull);

    const wire::HelloAck ack = wire::decode_hello_ack(wire::encode(wire::HelloAck{ false, "no runtime" }));
    REQUIRE_FALSE(ack.accepted);
    REQUIRE(ack.reason == "no runtime");

    const wire::HelloAck ok = wire::decode_hello_ack(wire::encode(wire::HelloAck{ true, "" }));
    REQUIRE(ok.accepted);
    REQUIRE(ok.reason.empty());
}

TEST_CASE("OPEN_HAND encodes both hands and rejects others", "[remote_device_bridge][unit]")
{
    REQUIRE(wire::encode(wire::OpenHand{ XR_HAND_LEFT_EXT }) == std::vector<uint8_t>{ 0 });
    REQUIRE(wire::encode(wire::OpenHand{ XR_HAND_RIGHT_EXT }) == std::vector<uint8_t>{ 1 });
    REQUIRE(wire::decode_open_hand({ 0 }).hand == XR_HAND_LEFT_EXT);
    REQUIRE(wire::decode_open_hand({ 1 }).hand == XR_HAND_RIGHT_EXT);
    REQUIRE_THROWS_AS(wire::decode_open_hand({ 2 }), wire::DecodeError);
    REQUIRE_THROWS_AS(wire::decode_open_hand({}), wire::DecodeError);
    REQUIRE_THROWS_AS(wire::decode_open_hand({ 0, 0 }), wire::DecodeError);
}

TEST_CASE("OPEN_SCHEMA round-trips SchemaPusherConfig", "[remote_device_bridge][unit]")
{
    core::SchemaPusherConfig config{ .collection_id = "manus_sensors_left",
                                     .max_flatbuffer_size = 4096,
                                     .tensor_identifier = "joint_state",
                                     .localized_name = "Manus Sensors Left",
                                     .app_name = "" };
    const core::SchemaPusherConfig decoded = wire::decode_open_schema(wire::encode(config));
    REQUIRE(decoded.collection_id == config.collection_id);
    REQUIRE(decoded.max_flatbuffer_size == 4096);
    REQUIRE(decoded.tensor_identifier == config.tensor_identifier);
    REQUIRE(decoded.localized_name == config.localized_name);
    REQUIRE(decoded.app_name.empty());

    std::vector<uint8_t> truncated = wire::encode(config);
    truncated.pop_back();
    REQUIRE_THROWS_AS(wire::decode_open_schema(truncated), wire::DecodeError);
}

TEST_CASE("OPEN_WRIST round-trips WristTrackingSourceConfig", "[remote_device_bridge][unit]")
{
    core::WristTrackingSourceConfig config;
    config.mode = core::WristTrackingSourceMode::Controller;
    config.left_aim_to_wrist = make_pose(0.1f);
    config.right_aim_to_wrist = make_pose(0.7f);

    const std::vector<uint8_t> bytes = wire::encode(config);
    REQUIRE(bytes.size() == wire::kWristConfigPayloadSize);
    const core::WristTrackingSourceConfig decoded = wire::decode_open_wrist(bytes);
    REQUIRE(decoded.mode == core::WristTrackingSourceMode::Controller);
    REQUIRE(same_pose(decoded.left_aim_to_wrist, config.left_aim_to_wrist));
    REQUIRE(same_pose(decoded.right_aim_to_wrist, config.right_aim_to_wrist));

    REQUIRE(wire::decode_open_wrist(wire::encode(core::WristTrackingSourceConfig{})).mode ==
            core::WristTrackingSourceMode::Auto);
    std::vector<uint8_t> bad_mode = bytes;
    bad_mode[0] = 7;
    REQUIRE_THROWS_AS(wire::decode_open_wrist(bad_mode), wire::DecodeError);
}

TEST_CASE("HAND_SAMPLE is a bit-exact copy of 26 joints", "[remote_device_bridge][unit]")
{
    wire::HandSample sample;
    sample.sample_time_ns = 987654321012345LL;
    for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i)
    {
        sample.joints[static_cast<size_t>(i)].locationFlags = 0xF000000000000000ULL | static_cast<uint64_t>(i);
        sample.joints[static_cast<size_t>(i)].pose = make_pose(0.01f * static_cast<float>(i + 1));
        sample.joints[static_cast<size_t>(i)].radius = 0.001f * static_cast<float>(i);
    }

    const std::vector<uint8_t> bytes = hand_payload(sample);
    REQUIRE(bytes.size() == wire::kHandSamplePayloadSize);

    const wire::HandSample decoded = wire::decode_hand_sample(bytes);
    REQUIRE(decoded.sample_time_ns == sample.sample_time_ns);
    for (size_t i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i)
    {
        REQUIRE(decoded.joints[i].locationFlags == sample.joints[i].locationFlags);
        REQUIRE(same_pose(decoded.joints[i].pose, sample.joints[i].pose));
        REQUIRE(std::memcmp(&decoded.joints[i].radius, &sample.joints[i].radius, sizeof(float)) == 0);
    }

    std::vector<uint8_t> truncated(bytes.begin(), bytes.end() - 1);
    REQUIRE_THROWS_AS(wire::decode_hand_sample(truncated), wire::DecodeError);
    std::vector<uint8_t> extended = bytes;
    extended.push_back(0);
    REQUIRE_THROWS_AS(wire::decode_hand_sample(extended), wire::DecodeError);
}

TEST_CASE("SCHEMA_SAMPLE round-trips bytes and timestamps", "[remote_device_bridge][unit]")
{
    wire::SchemaSample sample;
    sample.sample_time_local_common_clock_ns = 111;
    sample.sample_time_raw_device_clock_ns = 222;
    sample.bytes = { 0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01 };

    const wire::SchemaSample decoded = wire::decode_schema_sample(schema_payload(sample));
    REQUIRE(decoded.sample_time_local_common_clock_ns == 111);
    REQUIRE(decoded.sample_time_raw_device_clock_ns == 222);
    REQUIRE(decoded.bytes == sample.bytes);

    const wire::SchemaSample empty = wire::decode_schema_sample(schema_payload(wire::SchemaSample{}));
    REQUIRE(empty.bytes.empty());

    std::vector<uint8_t> lying = schema_payload(sample);
    lying[16] = 0xFF; // declared size larger than the remaining bytes
    REQUIRE_THROWS_AS(wire::decode_schema_sample(lying), wire::DecodeError);
}

TEST_CASE("WRIST_SAMPLE round-trips", "[remote_device_bridge][unit]")
{
    wire::WristSample sample;
    sample.sample_time_ns = -5;
    sample.is_left = false;
    sample.sample.valid = true;
    sample.sample.tracked = false;
    sample.sample.pose = make_pose(0.3f);

    const std::vector<uint8_t> bytes = wire::encode(sample);
    REQUIRE(bytes.size() == wire::kWristSamplePayloadSize);
    const wire::WristSample decoded = wire::decode_wrist_sample(bytes);
    REQUIRE(decoded.sample_time_ns == -5);
    REQUIRE_FALSE(decoded.is_left);
    REQUIRE(decoded.sample.valid);
    REQUIRE_FALSE(decoded.sample.tracked);
    REQUIRE(same_pose(decoded.sample.pose, sample.sample.pose));

    std::vector<uint8_t> bad_bool = bytes;
    bad_bool[9] = 2;
    REQUIRE_THROWS_AS(wire::decode_wrist_sample(bad_bool), wire::DecodeError);
}

TEST_CASE("ERROR carries a message and empty payloads decode", "[remote_device_bridge][unit]")
{
    REQUIRE(wire::decode_error(wire::encode(wire::Error{ "boom" })).message == "boom");
    REQUIRE(wire::decode_error(wire::encode(wire::Error{ "" })).message.empty());
    REQUIRE_THROWS_AS(wire::decode_error({ 4, 0, 0, 0, 'a' }), wire::DecodeError);
    REQUIRE(std::string(wire::to_string(wire::FrameType::Heartbeat)) == "HEARTBEAT");
}

TEST_CASE("PING and PONG round-trip and PONG t3 is stamped at send time", "[remote_device_bridge][unit]")
{
    const std::vector<uint8_t> ping = wire::encode(wire::Ping{ -123456789012345LL });
    REQUIRE(ping.size() == 8);
    REQUIRE(wire::decode_ping(ping).t1_ns == -123456789012345LL);
    REQUIRE_THROWS_AS(wire::decode_ping({ 1, 2, 3 }), wire::DecodeError);

    const std::vector<uint8_t> payload = wire::encode(wire::Pong{ 1, 2, 3 });
    REQUIRE(payload.size() == 24);
    const wire::Pong pong = wire::decode_pong(payload);
    REQUIRE(pong.t1_ns == 1);
    REQUIRE(pong.t2_ns == 2);
    REQUIRE(pong.t3_ns == 3);
    REQUIRE_THROWS_AS(wire::decode_pong(std::vector<uint8_t>(25, 0)), wire::DecodeError);

    std::vector<uint8_t> frame = wire::encode_frame(wire::FrameType::Pong, wire::kSessionChannelId, payload);
    wire::stamp_header(frame.data(), 7, 0x0102030405060708LL);
    const wire::Frame split = wire::split_frame(frame);
    REQUIRE(split.header.type == wire::FrameType::Pong);
    REQUIRE(split.header.t_send_monotonic_ns == 0x0102030405060708LL);
    const wire::Pong stamped = wire::decode_pong(split.payload);
    REQUIRE(stamped.t1_ns == 1);
    REQUIRE(stamped.t2_ns == 2);
    REQUIRE(stamped.t3_ns == 0x0102030405060708LL);

    // stamp_header leaves the payload of every other frame type alone.
    std::vector<uint8_t> ping_frame = wire::encode_frame(wire::FrameType::Ping, wire::kSessionChannelId, ping);
    wire::stamp_header(ping_frame.data(), 8, 42);
    REQUIRE(wire::decode_ping(wire::split_frame(ping_frame).payload).t1_ns == -123456789012345LL);
    REQUIRE(std::string(wire::to_string(wire::FrameType::Ping)) == "PING");
    REQUIRE(std::string(wire::to_string(wire::FrameType::Pong)) == "PONG");
}

TEST_CASE("Contiguous frame builders match header + payload encoder byte for byte", "[remote_device_bridge][unit]")
{
    SECTION("SCHEMA_SAMPLE against a hand-built golden frame")
    {
        const std::vector<uint8_t> bytes = { 0xDE, 0xAD };
        std::vector<uint8_t> frame =
            wire::encode_schema_sample_frame(3, bytes.data(), bytes.size(), 0x0102030405060708LL, -1);
        // clang-format off
        const std::vector<uint8_t> golden = {
            0x52, 0x50, 0x53, 0x30,                         // magic "RPS0"
            0x09, 0x00,                                     // type SCHEMA_SAMPLE, version 0
            0x03, 0x00,                                     // channel 3
            0x00, 0x00, 0x00, 0x00,                         // seq (unstamped)
            0x16, 0x00, 0x00, 0x00,                         // payload_size 22
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // t_send (unstamped)
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // sample_time_local_common_clock_ns
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // sample_time_raw_device_clock_ns = -1
            0x02, 0x00, 0x00, 0x00,                         // byte count
            0xDE, 0xAD,
        };
        // clang-format on
        REQUIRE(frame == golden);

        wire::stamp_header(frame.data(), 0x11223344u, 0x0807060504030201LL);
        REQUIRE(std::vector<uint8_t>(frame.begin() + 8, frame.begin() + 12) ==
                std::vector<uint8_t>{ 0x44, 0x33, 0x22, 0x11 });
        REQUIRE(std::vector<uint8_t>(frame.begin() + 16, frame.begin() + 24) ==
                std::vector<uint8_t>{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 });
        REQUIRE(std::equal(frame.begin() + 24, frame.end(), golden.begin() + 24)); // payload untouched

        const wire::Frame split = wire::split_frame(frame);
        REQUIRE(split.header.type == wire::FrameType::SchemaSample);
        REQUIRE(split.header.channel_id == 3);
        REQUIRE(split.header.seq == 0x11223344u);
        REQUIRE(split.header.t_send_monotonic_ns == 0x0807060504030201LL);
        REQUIRE(split.payload == std::vector<uint8_t>(golden.begin() + 24, golden.end()));
        wire::SchemaSample decoded = wire::decode_schema_sample(split.payload);
        REQUIRE(decoded.bytes == bytes);
        REQUIRE(decoded.sample_time_raw_device_clock_ns == -1);
    }
    SECTION("HAND_SAMPLE against the reference payload encoder")
    {
        wire::HandSample sample;
        sample.sample_time_ns = -42;
        for (size_t i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i)
        {
            sample.joints[i].locationFlags = 0xA000000000000000ULL | i;
            sample.joints[i].pose = make_pose(0.02f * static_cast<float>(i + 1));
            sample.joints[i].radius = 0.001f * static_cast<float>(i);
        }
        const std::vector<uint8_t> payload = reference_hand_payload(sample);
        REQUIRE(payload.size() == wire::kHandSamplePayloadSize);
        const std::vector<uint8_t> frame = wire::encode_hand_sample_frame(0x0102, sample.joints.data(), -42);
        REQUIRE(frame.size() == wire::kHeaderSize + wire::kHandSamplePayloadSize);
        const wire::FrameHeader header = wire::decode_header(frame.data());
        REQUIRE(header.type == wire::FrameType::HandSample);
        REQUIRE(header.channel_id == 0x0102);
        REQUIRE(header.payload_size == wire::kHandSamplePayloadSize);
        REQUIRE(std::equal(frame.begin() + 24, frame.end(), payload.begin(), payload.end()));
        REQUIRE(wire::encode_frame(wire::FrameType::HandSample, 0x0102, payload) == frame);
    }
    SECTION("split_frame rejects short or inconsistent messages")
    {
        REQUIRE_THROWS_AS(wire::split_frame(std::vector<uint8_t>(10, 0)), wire::DecodeError);
        std::vector<uint8_t> frame = wire::encode_frame(wire::FrameType::Heartbeat, 0, {});
        REQUIRE(frame.size() == wire::kHeaderSize);
        REQUIRE(wire::split_frame(frame).payload.empty());
        frame.push_back(0);
        REQUIRE_THROWS_AS(wire::split_frame(frame), wire::DecodeError);
    }
    SECTION("payloads above the limit are refused at build time")
    {
        std::vector<uint8_t> too_big(wire::kMaxPayloadSize + 1);
        REQUIRE_THROWS_AS(
            wire::encode_schema_sample_frame(1, too_big.data(), too_big.size(), 0, 0), std::invalid_argument);
    }
}
