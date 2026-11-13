/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OutputRateSelector.h"
#include "TestHarness.h"

#include <cstdint>

namespace
{

constexpr uint64_t kStartNs = 1000000000ULL;

// Feeds `frames` frames spaced `inputPeriodNs` apart (plus jitterNs(i)) and returns how many are sent.
template <typename Jitter>
uint64_t countSent(uint32_t fps, uint64_t frames, uint64_t inputPeriodNs, Jitter jitterNs)
{
    argus_sender::OutputRateSelector selector(fps);
    uint64_t sent = 0;
    for (uint64_t i = 0; i < frames; ++i)
    {
        const uint64_t frameNs = kStartNs + i * inputPeriodNs + static_cast<uint64_t>(jitterNs(i));
        if (selector.shouldSkip(frameNs))
            continue;
        selector.sent(frameNs);
        ++sent;
    }
    return sent;
}

uint64_t noJitter(uint64_t)
{
    return 0;
}

} // namespace

int main()
{
    TestContext test("output_rate_selector");

    {
        argus_sender::OutputRateSelector selector(30);
        TEST_CHECK(test, !selector.shouldSkip(kStartNs));
    }

    // A 60 Hz source at 30 fps sends every second frame.
    {
        argus_sender::OutputRateSelector selector(30);
        for (uint64_t i = 0; i < 600; ++i)
        {
            const uint64_t frameNs = kStartNs + i * 16666667ULL;
            const bool skip = selector.shouldSkip(frameNs);
            TEST_CHECK_EQ(test, skip, (i % 2) == 1);
            if (!skip)
                selector.sent(frameNs);
        }
    }

    // A 30 Hz source at 30 fps loses nothing to timing jitter: worst case alternating -J/+J gives gaps of period-2J.
    {
        const auto alternating = [](int64_t jitter)
        { return [jitter](uint64_t i) { return jitter + ((i % 2) ? jitter : -jitter); }; };
        TEST_CHECK_EQ(test, countSent(30, 300, 33333333ULL, alternating(2000000)), 300ULL);
        TEST_CHECK_EQ(test, countSent(30, 300, 33333333ULL, alternating(3000000)), 300ULL);
    }

    // fps 0 never skips, even for identical timestamps.
    {
        argus_sender::OutputRateSelector selector(0);
        selector.sent(kStartNs);
        TEST_CHECK(test, !selector.shouldSkip(kStartNs));
        TEST_CHECK(test, !selector.shouldSkip(kStartNs + 1));
        TEST_CHECK_EQ(test, countSent(0, 100, 1000000ULL, noJitter), 100ULL);
    }

    // A frame earlier than the last one sent (clock stepped back) is not skipped.
    {
        argus_sender::OutputRateSelector selector(30);
        selector.sent(kStartNs + 500000000ULL);
        TEST_CHECK(test, !selector.shouldSkip(kStartNs));
        TEST_CHECK(test, !selector.shouldSkip(kStartNs + 500000000ULL - 1));
    }

    // A gap longer than a period restarts the grid at the frame that ended it.
    {
        argus_sender::OutputRateSelector selector(30);
        selector.sent(kStartNs);
        const uint64_t resumeNs = kStartNs + 200000000ULL;
        TEST_CHECK(test, !selector.shouldSkip(resumeNs));
        selector.sent(resumeNs);
        TEST_CHECK(test, selector.shouldSkip(resumeNs + 10000000ULL));
        TEST_CHECK(test, !selector.shouldSkip(resumeNs + 33333333ULL));
        selector.sent(resumeNs + 33333333ULL);
        TEST_CHECK(test, selector.shouldSkip(resumeNs + 33333333ULL + 10000000ULL));
    }

    // After the clock steps back, the next frame is sent and the grid restarts from it.
    {
        argus_sender::OutputRateSelector selector(30);
        selector.sent(kStartNs + 1000000000ULL);
        const uint64_t backNs = kStartNs + 100000000ULL;
        TEST_CHECK(test, !selector.shouldSkip(backNs));
        selector.sent(backNs);
        TEST_CHECK(test, selector.shouldSkip(backNs + 10000000ULL));
        TEST_CHECK(test, !selector.shouldSkip(backNs + 33333333ULL));
    }

    // 24 fps from a 30 Hz source keeps about 24 of every 30 frames.
    {
        const uint64_t sent = countSent(24, 30, 33333333ULL, noJitter);
        TEST_CHECK(test, sent >= 23 && sent <= 25);
        const uint64_t sentLong = countSent(24, 300, 33333333ULL, noJitter);
        TEST_CHECK(test, sentLong >= 230 && sentLong <= 250);
    }

    return test.finish();
}
