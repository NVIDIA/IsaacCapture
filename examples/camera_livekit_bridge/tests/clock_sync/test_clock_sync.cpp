// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <livekit_publisher/clock_sync.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace camera_bridge;

namespace
{

int g_failures = 0;

#define CHECK(cond)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                       \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

using Bytes = std::vector<std::uint8_t>;

// sequence 7, t1 123456789 (0x075BCD15)
const Bytes kRequest = { 'Q', 0x07, 0x00, 0x00, 0x00, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00 };

// t2 = 0x0102030405060708, t3 = 1000000000000 (0x000000E8D4A51000)
const Bytes kReply = { 'R',  0x07, 0x00, 0x00, 0x00, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00, 0x08, 0x07,
                       0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00, 0x10, 0xA5, 0xD4, 0xE8, 0x00, 0x00, 0x00 };

void test_parse_request()
{
    CHECK(kRequest.size() == kClockSyncRequestSize);
    const auto r = parse_clock_sync_request(kRequest);
    CHECK(r.has_value());
    if (r)
    {
        CHECK(r->sequence == 7);
        CHECK(r->t1 == 123456789);
    }

    // High bits: negative t1 and a sequence with the top byte set.
    Bytes high = { 'Q', 0x78, 0x56, 0x34, 0x82, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
    const auto h = parse_clock_sync_request(high);
    CHECK(h.has_value());
    if (h)
    {
        CHECK(h->sequence == 0x82345678u);
        CHECK(h->t1 == -1);
    }
}

void test_parse_request_rejects()
{
    CHECK(!parse_clock_sync_request({}).has_value());
    Bytes short_req(kRequest.begin(), kRequest.end() - 1);
    CHECK(!parse_clock_sync_request(short_req).has_value());
    Bytes long_req = kRequest;
    long_req.push_back(0);
    CHECK(!parse_clock_sync_request(long_req).has_value());
    Bytes wrong_tag = kRequest;
    wrong_tag[0] = 'R';
    CHECK(!parse_clock_sync_request(wrong_tag).has_value());
    wrong_tag[0] = 'q';
    CHECK(!parse_clock_sync_request(wrong_tag).has_value());
    // A reply-sized buffer is not a request.
    CHECK(!parse_clock_sync_request(kReply).has_value());
}

void test_build_reply()
{
    const Bytes reply = build_clock_sync_reply(ClockSyncRequest{ 7, 123456789 }, 0x0102030405060708LL, 1000000000000LL);
    CHECK(reply.size() == kClockSyncReplySize);
    CHECK(reply == kReply);

    const auto parsed = parse_clock_sync_reply(reply);
    CHECK(parsed.has_value());
    if (parsed)
    {
        CHECK(parsed->sequence == 7);
        CHECK(parsed->t1 == 123456789);
        CHECK(parsed->t2 == 0x0102030405060708LL);
        CHECK(parsed->t3 == 1000000000000LL);
    }
}

void test_set_t3()
{
    Bytes reply = build_clock_sync_reply(ClockSyncRequest{ 7, 123456789 }, 0x0102030405060708LL, 0);
    const Bytes before = reply;
    set_clock_sync_reply_t3(reply, 1000000000000LL);
    CHECK(reply == kReply);
    // Only the eight bytes at the t3 offset differ.
    for (std::size_t i = 0; i < reply.size(); ++i)
        if (i < kClockSyncReplyT3Offset || i >= kClockSyncReplyT3Offset + 8)
            CHECK(reply[i] == before[i]);
    CHECK(kClockSyncReplyT3Offset == 21);
}

void test_request_round_trip_and_rejects()
{
    CHECK(build_clock_sync_request(ClockSyncRequest{ 7, 123456789 }) == kRequest);
    CHECK(!parse_clock_sync_reply(kRequest).has_value());
    Bytes bad = kReply;
    bad[0] = 'Q';
    CHECK(!parse_clock_sync_reply(bad).has_value());
    bad = kReply;
    bad.pop_back();
    CHECK(!parse_clock_sync_reply(bad).has_value());
}

} // namespace

int main()
{
    test_parse_request();
    test_parse_request_rejects();
    test_build_reply();
    test_set_t3();
    test_request_round_trip_and_rejects();
    if (g_failures != 0)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("clock_sync tests passed\n");
    return 0;
}
