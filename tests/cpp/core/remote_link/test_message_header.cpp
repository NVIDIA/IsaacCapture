// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <remote_link/message_header.hpp>
#include <remote_link/messages.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace core;

TEST_CASE("RemoteLink: header round-trips and has the documented layout", "[remote_link][unit]")
{
    LinkMessageHeader header;
    header.type = 34;
    header.channel = 0x0102;
    header.seq = 0x03040506;
    header.payload_size = 0x00070809;
    header.t_send_ns = -1234567890123;
    std::array<uint8_t, kHeaderSize> bytes{};
    encode_header(header, bytes.data());

    const std::array<uint8_t, 8> prefix{ 0x52, 0x54, 0x53, 0x30, 34, 0, 0x02, 0x01 }; // "RTS0", type, version, channel
    REQUIRE(std::equal(prefix.begin(), prefix.end(), bytes.begin()));
    REQUIRE(bytes[8] == 0x06);
    REQUIRE(bytes[12] == 0x09);

    const LinkMessageHeader decoded = decode_header(bytes.data());
    REQUIRE(decoded.type == 34);
    REQUIRE(decoded.channel == 0x0102);
    REQUIRE(decoded.seq == 0x03040506);
    REQUIRE(decoded.payload_size == 0x00070809);
    REQUIRE(decoded.t_send_ns == -1234567890123);
}

TEST_CASE("RemoteLink: decode_header rejects bad magic, version and size only", "[remote_link][unit]")
{
    std::array<uint8_t, kHeaderSize> bytes{};
    LinkMessageHeader header;
    header.type = 250; // unknown types are not an error
    encode_header(header, bytes.data());
    REQUIRE(decode_header(bytes.data()).type == 250);

    SECTION("magic")
    {
        bytes[0] ^= 0xFF;
        REQUIRE_THROWS_AS(decode_header(bytes.data()), LinkMessageDecodeError);
    }
    SECTION("version")
    {
        bytes[5] = 1;
        REQUIRE_THROWS_AS(decode_header(bytes.data()), LinkMessageDecodeError);
    }
    SECTION("payload size")
    {
        header.payload_size = kMaxPayloadSize;
        encode_header(header, bytes.data());
        REQUIRE(decode_header(bytes.data()).payload_size == kMaxPayloadSize);
        header.payload_size = kMaxPayloadSize + 1;
        encode_header(header, bytes.data());
        REQUIRE_THROWS_AS(decode_header(bytes.data()), LinkMessageDecodeError);
    }
}

TEST_CASE("RemoteLink: encode_link_message and stamp_header", "[remote_link][unit]")
{
    const std::vector<uint8_t> payload{ 1, 2, 3 };
    std::vector<uint8_t> message = encode_link_message(static_cast<uint8_t>(LinkMessageType::Status), payload);
    REQUIRE(message.size() == kHeaderSize + 3);
    LinkMessageHeader header = decode_header(message.data());
    REQUIRE(header.type == static_cast<uint8_t>(LinkMessageType::Status));
    REQUIRE(header.seq == 0);
    REQUIRE(header.t_send_ns == 0);
    REQUIRE(header.payload_size == 3);
    REQUIRE(std::vector<uint8_t>(message.begin() + kHeaderSize, message.end()) == payload);

    stamp_header(message.data(), 77, 123456789);
    header = decode_header(message.data());
    REQUIRE(header.seq == 77);
    REQUIRE(header.t_send_ns == 123456789);
    REQUIRE(header.payload_size == 3);

    REQUIRE(encode_link_message(11, {}).size() == kHeaderSize);
    const std::vector<uint8_t> too_big(kMaxPayloadSize + 1);
    REQUIRE_THROWS_AS(encode_link_message(34, too_big), std::invalid_argument);
}

TEST_CASE("RemoteLink: PONG and SAMPLES payload codecs", "[remote_link][unit]")
{
    const std::vector<uint8_t> pong = encode_pong(PongPayload{ -5, 1'000'000'000'000 });
    REQUIRE(pong.size() == kPongPayloadSize);
    const PongPayload decoded = decode_pong(pong);
    REQUIRE(decoded.t1_ns == -5);
    REQUIRE(decoded.t2_ns == 1'000'000'000'000);
    REQUIRE_THROWS_AS(decode_pong(std::vector<uint8_t>(15)), LinkMessageDecodeError);
    REQUIRE_THROWS_AS(decode_pong(std::vector<uint8_t>(17)), LinkMessageDecodeError);

    const std::vector<uint8_t> records{ 9, 8, 7 };
    const std::vector<uint8_t> samples = encode_samples_payload(SamplesPrefix{ 0xFFFF'FFFF'0000'0001ull, 42 }, records);
    REQUIRE(samples.size() == kSamplesPrefixSize + records.size());
    const SamplesPrefix prefix = decode_samples_prefix(samples);
    REQUIRE(prefix.tick_seq == 0xFFFF'FFFF'0000'0001ull);
    REQUIRE(prefix.tick_time_ns == 42);
    REQUIRE(std::vector<uint8_t>(samples.begin() + kSamplesPrefixSize, samples.end()) == records);
    REQUIRE(decode_samples_prefix(std::vector<uint8_t>(kSamplesPrefixSize)).tick_seq == 0);
    REQUIRE_THROWS_AS(decode_samples_prefix(std::vector<uint8_t>(kSamplesPrefixSize - 1)), LinkMessageDecodeError);
}

TEST_CASE("RemoteLink: message type ids and names", "[remote_link][unit]")
{
    REQUIRE(static_cast<int>(LinkMessageType::Hello) == 1);
    REQUIRE(static_cast<int>(LinkMessageType::HelloAck) == 2);
    REQUIRE(static_cast<int>(LinkMessageType::Heartbeat) == 11);
    REQUIRE(static_cast<int>(LinkMessageType::Error) == 12);
    REQUIRE(static_cast<int>(LinkMessageType::Ping) == 13);
    REQUIRE(static_cast<int>(LinkMessageType::Pong) == 14);
    REQUIRE(static_cast<int>(LinkMessageType::Status) == 32);
    REQUIRE(static_cast<int>(LinkMessageType::Setup) == 33);
    REQUIRE(static_cast<int>(LinkMessageType::Samples) == 34);
    REQUIRE(static_cast<int>(LinkMessageType::Ready) == 35);
    REQUIRE(std::string(to_string(LinkMessageType::Samples)) == "SAMPLES");
    REQUIRE(std::string(to_string(static_cast<LinkMessageType>(200))) == "UNKNOWN");
}
