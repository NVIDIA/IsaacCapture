// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <video_stage/clock_sync.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using plugins::sensing::build_clock_sync_reply;
using plugins::sensing::ClockSyncRequest;
using plugins::sensing::kClockSyncReplySize;
using plugins::sensing::kClockSyncRequestSize;
using plugins::sensing::monotonic_ns;
using plugins::sensing::parse_clock_sync_request;
using plugins::sensing::set_clock_sync_reply_t3;

namespace
{

using Bytes = std::vector<std::uint8_t>;

Bytes from_hex(const std::string& hex)
{
    Bytes out;
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2)
        out.push_back(static_cast<std::uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    return out;
}

// Golden bytes produced by egocap_web.livekit_clock_sync.encode_request and
// tools/livekit_orin_emulator.build_sync_reply for the inputs named beside each constant.
// sequence 1, t1 1, t2 2, t3 3
const char* const kSimpleRequest = "51010000000100000000000000";
const char* const kSimpleReply = "5201000000010000000000000002000000000000000300000000000000";
// sequence 0xFFFFFFFF, t1 -1234567890123, t2 INT64_MAX, t3 INT64_MIN
const char* const kExtremeRequest = "51ffffffff35fb048ee0feffff";
const char* const kExtremeReply = "52ffffffff35fb048ee0feffffffffffffffffff7f0000000000000080";

} // namespace

TEST_CASE("parse_clock_sync_request accepts a well-formed request", "[clock_sync]")
{
    auto request = parse_clock_sync_request(from_hex(kSimpleRequest));
    REQUIRE(request.has_value());
    CHECK(request->sequence == 1);
    CHECK(request->t1 == 1);

    request = parse_clock_sync_request(from_hex(kExtremeRequest));
    REQUIRE(request.has_value());
    CHECK(request->sequence == 0xFFFFFFFFu);
    CHECK(request->t1 == -1234567890123LL);
}

TEST_CASE("parse_clock_sync_request rejects anything else", "[clock_sync]")
{
    for (std::size_t size : { std::size_t{ 0 }, std::size_t{ 12 }, std::size_t{ 14 }, std::size_t{ 29 } })
    {
        Bytes data(size, 0);
        if (size > 0)
            data[0] = 'Q';
        CHECK_FALSE(parse_clock_sync_request(data).has_value());
    }

    Bytes wrong_kind = from_hex(kSimpleRequest);
    wrong_kind[0] = 'X';
    CHECK_FALSE(parse_clock_sync_request(wrong_kind).has_value());

    CHECK_FALSE(parse_clock_sync_request(from_hex(kSimpleReply)).has_value());

    Bytes reply_kind_at_request_size = from_hex(kSimpleRequest);
    reply_kind_at_request_size[0] = 'R';
    CHECK_FALSE(parse_clock_sync_request(reply_kind_at_request_size).has_value());
}

TEST_CASE("build_clock_sync_reply lays out the documented fields", "[clock_sync]")
{
    const Bytes reply = build_clock_sync_reply(ClockSyncRequest{ 1, 1 }, 2, 3);
    REQUIRE(reply.size() == kClockSyncReplySize);
    CHECK(reply[0] == 'R');
    CHECK(reply == from_hex(kSimpleReply));

    // Explicit offsets, little-endian.
    CHECK(reply[1] == 1);
    CHECK(reply[5] == 1);
    CHECK(reply[13] == 2);
    CHECK(reply[21] == 3);
}

TEST_CASE("build_clock_sync_reply handles large and negative values", "[clock_sync]")
{
    const ClockSyncRequest request{ 0xFFFFFFFFu, -1234567890123LL };
    const Bytes reply = build_clock_sync_reply(
        request, std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::min());
    CHECK(reply == from_hex(kExtremeReply));
}

TEST_CASE("clock sync request and reply round trip through parse", "[clock_sync]")
{
    const Bytes request_bytes = from_hex(kExtremeRequest);
    REQUIRE(request_bytes.size() == kClockSyncRequestSize);
    const auto request = parse_clock_sync_request(request_bytes);
    REQUIRE(request.has_value());
    const Bytes reply = build_clock_sync_reply(
        *request, std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::min());
    CHECK(reply == from_hex(kExtremeReply));
}

TEST_CASE("set_clock_sync_reply_t3 changes only the t3 bytes", "[clock_sync]")
{
    const Bytes before = build_clock_sync_reply(ClockSyncRequest{ 7, -5 }, 11, 13);
    Bytes after = before;
    set_clock_sync_reply_t3(after, 0x0102030405060708LL);

    REQUIRE(after.size() == before.size());
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        if (i >= 21 && i <= 28)
            CHECK(after[i] == static_cast<std::uint8_t>(0x08 - (i - 21)));
        else
            CHECK(after[i] == before[i]);
    }
    CHECK(after == build_clock_sync_reply(ClockSyncRequest{ 7, -5 }, 11, 0x0102030405060708LL));
}

TEST_CASE("monotonic_ns does not go backwards", "[clock_sync]")
{
    const std::int64_t a = monotonic_ns();
    const std::int64_t b = monotonic_ns();
    CHECK(a > 0);
    CHECK(b >= a);
}
