/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ClockExchange.h"
#include "TestHarness.h"

#include <cstdint>
#include <ctime>
#include <string>

namespace
{

using argus_sender::buildClockReply;
using argus_sender::kClockReplyBytes;
using argus_sender::kClockRequestBytes;
using argus_sender::parseClockRequest;

void putLe(uint8_t* out, uint64_t value, int bytes)
{
    for (int i = 0; i < bytes; ++i)
        out[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xFF);
}

std::string makeRequest(uint32_t seq, int64_t t1)
{
    uint8_t bytes[kClockRequestBytes];
    bytes[0] = 'Q';
    putLe(bytes + 1, seq, 4);
    putLe(bytes + 5, static_cast<uint64_t>(t1), 8);
    return std::string(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

const uint8_t* bytesOf(const std::string& s)
{
    return reinterpret_cast<const uint8_t*>(s.data());
}

bool parse(const std::string& request, uint32_t* seq, int64_t* t1)
{
    return parseClockRequest(bytesOf(request), request.size(), seq, t1);
}

int64_t posixMonotonicNs()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

} // namespace

int main()
{
    TestContext test("clock_exchange");

    TEST_CHECK_EQ(test, kClockRequestBytes, static_cast<size_t>(13));
    TEST_CHECK_EQ(test, kClockReplyBytes, static_cast<size_t>(29));

    // Valid requests.
    {
        uint32_t seq = 0;
        int64_t t1 = 0;
        TEST_CHECK(test, parse(makeRequest(7, 123456789012345LL), &seq, &t1));
        TEST_CHECK_EQ(test, seq, 7U);
        TEST_CHECK_EQ(test, t1, 123456789012345LL);

        TEST_CHECK(test, parse(makeRequest(0xFFFFFFFFU, -42), &seq, &t1));
        TEST_CHECK_EQ(test, seq, 0xFFFFFFFFU);
        TEST_CHECK_EQ(test, t1, static_cast<int64_t>(-42));

        TEST_CHECK(test, parse(makeRequest(0, INT64_MIN), &seq, &t1));
        TEST_CHECK_EQ(test, t1, INT64_MIN);
        TEST_CHECK(test, parse(makeRequest(1, INT64_MAX), &seq, &t1));
        TEST_CHECK_EQ(test, t1, INT64_MAX);
    }

    // Explicit byte layout of a request.
    {
        const uint8_t raw[13] = { 'Q', 0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01 };
        uint32_t seq = 0;
        int64_t t1 = 0;
        TEST_CHECK(test, parseClockRequest(raw, sizeof(raw), &seq, &t1));
        TEST_CHECK_EQ(test, seq, 0x01020304U);
        TEST_CHECK_EQ(test, t1, 0x0102030405060708LL);
    }

    // Rejections.
    {
        const std::string good = makeRequest(5, 99);
        uint32_t seq = 111;
        int64_t t1 = 222;
        TEST_CHECK(test, !parseClockRequest(bytesOf(good), 12, &seq, &t1));
        TEST_CHECK(test, !parseClockRequest(bytesOf(good), 0, &seq, &t1));
        const std::string longer = good + 'x';
        TEST_CHECK_EQ(test, longer.size(), static_cast<size_t>(14));
        TEST_CHECK(test, !parseClockRequest(bytesOf(longer), longer.size(), &seq, &t1));
        std::string wrongTag = good;
        wrongTag[0] = 'X';
        TEST_CHECK(test, !parse(wrongTag, &seq, &t1));
        wrongTag[0] = 'q';
        TEST_CHECK(test, !parse(wrongTag, &seq, &t1));
        wrongTag[0] = 'R';
        TEST_CHECK(test, !parse(wrongTag, &seq, &t1));
        TEST_CHECK(test, !parseClockRequest(nullptr, kClockRequestBytes, &seq, &t1));
        TEST_CHECK(test, !parseClockRequest(nullptr, 0, &seq, &t1));
    }

    // Reply layout.
    {
        const int64_t t1 = -5;
        const int64_t t2 = 0x1122334455667788LL;
        const int64_t t3 = (1LL << 32) + 0x0A0B0C0DLL;
        const std::string reply = buildClockReply(0xA1B2C3D4U, t1, t2, t3);
        TEST_CHECK_EQ(test, reply.size(), kClockReplyBytes);
        const uint8_t* b = bytesOf(reply);
        TEST_CHECK_EQ(test, static_cast<int>(b[0]), static_cast<int>('R'));
        const uint8_t expected[28] = {
            0xD4, 0xC3, 0xB2, 0xA1, // seq
            0xFB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // t1 = -5
            0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // t2
            0x0D, 0x0C, 0x0B, 0x0A, 0x01, 0x00, 0x00, 0x00 // t3 = 2^32 + 0x0A0B0C0D (bytes 4..7 = 01 00 00 00)
        };
        for (size_t i = 0; i < sizeof(expected); ++i)
            TEST_CHECK_EQ(test, static_cast<int>(b[1 + i]), static_cast<int>(expected[i]));
    }

    // Round trip.
    {
        const int64_t t1 = 1000000000000LL;
        const std::string request = makeRequest(31337, t1);
        uint32_t seq = 0;
        int64_t parsedT1 = 0;
        TEST_CHECK(test, parse(request, &seq, &parsedT1));
        const std::string reply = buildClockReply(seq, parsedT1, t1 + 100, t1 + 250);
        TEST_CHECK_EQ(test, reply.size(), kClockReplyBytes);
        // The echoed (seq, t1) in the reply equal the request's bytes 1..12.
        TEST_CHECK(test, reply.compare(1, 12, request, 1, 12) == 0);
        TEST_CHECK(test, reply[0] == 'R');
    }

    // Clock.
    {
        int64_t previous = argus_sender::monotonicClockNs();
        bool nonDecreasing = true;
        for (int i = 0; i < 1000; ++i)
        {
            const int64_t now = argus_sender::monotonicClockNs();
            if (now < previous)
                nonDecreasing = false;
            previous = now;
        }
        TEST_CHECK(test, nonDecreasing);

        const int64_t before = posixMonotonicNs();
        const int64_t ours = argus_sender::monotonicClockNs();
        const int64_t after = posixMonotonicNs();
        TEST_CHECK(test, before <= ours);
        TEST_CHECK(test, ours <= after);
        TEST_CHECK(test, ours > 0);
    }

    return test.finish();
}
