// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <realhand_ffg_glove/realhand_ffg_glove_protocol.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

using namespace plugins::realhand_ffg_glove;

namespace
{

void check(bool condition, const char* message)
{
    INFO(message);
    REQUIRE(condition);
}

template <typename T>
void append_little_endian(std::vector<uint8_t>& bytes, T value)
{
    std::array<uint8_t, sizeof(T)> encoded{};
    std::memcpy(encoded.data(), &value, sizeof(T));
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
}

std::optional<Frame> parse_all(const std::vector<uint8_t>& bytes)
{
    FrameParser parser;
    std::optional<Frame> result;
    for (uint8_t byte : bytes)
    {
        if (auto frame = parser.process(byte))
        {
            result = std::move(frame);
        }
    }
    return result;
}

TEST_CASE("pack_and_parse", "[realhand_ffg_protocol]")
{
    const auto encoded = pack_frame(Command::Position, { 1, 2, 3 });
    check(encoded == std::vector<uint8_t>({ 0x5D, 0x03, 0x03, 0x01, 0x02, 0x03, 0x69 }), "packed frame");
    const auto frame = parse_all(encoded);
    check(frame.has_value(), "parse packed frame");
    check(frame->command == Command::Position, "parsed command");
    check(frame->payload == std::vector<uint8_t>({ 1, 2, 3 }), "parsed payload");
}

TEST_CASE("bad_checksum_and_resync", "[realhand_ffg_protocol]")
{
    auto bad = pack_frame(Command::Version);
    bad.back() ^= 1;
    // Decimal 20104 is formatted by the SDK as major=2, minor=01, patch=04.
    const auto good = pack_frame(Command::Version, { 0x88, 0x4E, 0x00, 0x00, 0x01 });
    bad.insert(bad.end(), good.begin(), good.end());
    const auto frame = parse_all(bad);
    check(frame.has_value(), "parser resynchronizes after bad checksum");
    const auto version = decode_version(*frame);
    check(version.has_value(), "decode version");
    check(version->version == "2.1.4", "version formatting matches Python SDK");
    check(version->side == HandSide::Right, "right side decode");
}

TEST_CASE("float_positions", "[realhand_ffg_protocol]")
{
    std::vector<uint8_t> payload;
    for (std::size_t i = 0; i < kNumSensors; ++i)
    {
        append_little_endian(payload, static_cast<float>(i * 10));
    }
    const auto frame = parse_all(pack_frame(Command::Position, payload));
    check(frame.has_value(), "parse float position frame");
    const auto positions = decode_positions(*frame);
    check(positions.has_value(), "decode 21 float positions");
    check(std::abs((*positions)[9] - 1.5707963268f) < 1e-5f, "degrees converted to radians");

    payload.resize(payload.size() - sizeof(float));
    const auto short_frame = parse_all(pack_frame(Command::Position, payload));
    check(!decode_positions(*short_frame).has_value(), "reject wrong float sensor count");
}

TEST_CASE("a6_positions", "[realhand_ffg_protocol]")
{
    std::vector<uint8_t> payload;
    for (std::size_t i = 0; i < kNumSensors; ++i)
    {
        append_little_endian(payload, static_cast<int16_t>(i == 0 ? -9000 : 0));
    }
    const auto frame = parse_all(pack_frame(Command::A6Position, payload));
    const auto positions = decode_positions(*frame);
    check(positions.has_value(), "decode A6 int16 positions");
    check(std::abs((*positions)[0] + 1.5707963268f) < 1e-5f, "A6 hundredth-degrees converted to radians");

    const auto response = pack_frame(Command::A7Force, std::vector<uint8_t>(5 * sizeof(float), 0));
    const auto response_frame = parse_all(response);
    check(response_frame.has_value(), "parse A7 continuation frame");
    check(response_frame->command == Command::A7Force, "A7 continuation command");
    check(response_frame->payload.size() == 5 * sizeof(float), "A7 continuation has five force channels");
}

} // namespace
