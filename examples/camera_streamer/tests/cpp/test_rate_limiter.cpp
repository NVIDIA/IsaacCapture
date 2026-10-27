/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "MixedRateScheduler.h"
#include "StereoTiming.h"
#include "TestHarness.h"

#include <cstdint>
#include <vector>

namespace
{

uint64_t countOutputFrames(uint32_t targetFps, uint64_t inputFrames, uint64_t inputPeriodNs)
{
    argus_sender::StereoOutputRateLimiter limiter(targetFps);
    uint64_t outputFrames = 0;
    for (uint64_t i = 0; i < inputFrames; ++i)
    {
        argus_sender::FrameTiming timing;
        timing.hasSensorSofTimestampTsc = true;
        timing.sensorSofTimestampTscNs = 1000000000ULL + i * inputPeriodNs;
        if (limiter.shouldOutput(timing))
            ++outputFrames;
    }
    return outputFrames;
}

} // namespace

int main()
{
    TestContext test("rate_limiter");
    const uint64_t period60FpsNs = 16666667ULL;

    TEST_CHECK_EQ(test, countOutputFrames(0, 600, period60FpsNs), 600ULL);
    TEST_CHECK_EQ(test, countOutputFrames(60, 600, period60FpsNs), 600ULL);
    TEST_CHECK_EQ(test, countOutputFrames(30, 600, period60FpsNs), 300ULL);
    TEST_CHECK_EQ(test, countOutputFrames(45, 600, period60FpsNs), 450ULL);
    TEST_CHECK_EQ(test, countOutputFrames(45, 36000, period60FpsNs), 27000ULL);

    argus_sender::StereoOutputRateLimiter limiter(30);
    {
        argus_sender::StereoOutputRateLimiter jittered60Fps(60);
        uint64_t outputFrames = 0;
        for (uint64_t i = 0; i < 600; ++i)
        {
            argus_sender::FrameTiming jittered;
            jittered.hasSensorSofTimestampTsc = true;
            const int64_t jitterNs = i % 2 == 0 ? 500000 : -500000;
            jittered.sensorSofTimestampTscNs =
                static_cast<uint64_t>(1000000000LL + static_cast<int64_t>(i * period60FpsNs) + jitterNs);
            if (jittered60Fps.shouldOutput(jittered))
                ++outputFrames;
        }
        TEST_CHECK_EQ(test, outputFrames, 600ULL);
    }

    {
        argus_sender::StereoOutputRateLimiter baseLimiter(45);
        argus_sender::MixedRateScheduler scheduler({ 45, 30 });
        uint64_t bundles = 0;
        uint64_t lowerRateFrames = 0;
        for (uint64_t i = 0; i < 600; ++i)
        {
            argus_sender::FrameTiming baseTiming;
            baseTiming.hasSensorSofTimestampTsc = true;
            baseTiming.sensorSofTimestampTscNs = 1000000000ULL + i * period60FpsNs;
            if (!baseLimiter.shouldOutput(baseTiming))
                continue;
            const std::vector<bool> due = scheduler.next();
            ++bundles;
            lowerRateFrames += due[1] ? 1 : 0;
        }
        TEST_CHECK_EQ(test, bundles, 450ULL);
        TEST_CHECK_EQ(test, lowerRateFrames, 300ULL);
    }

    argus_sender::FrameTiming timing;
    timing.hasSensorSofTimestampTsc = true;
    timing.sensorSofTimestampTscNs = 1000000000ULL;
    TEST_CHECK(test, limiter.shouldOutput(timing));
    timing.sensorSofTimestampTscNs += period60FpsNs;
    TEST_CHECK(test, !limiter.shouldOutput(timing));

    timing.sensorSofTimestampTscNs = 500000000ULL;
    TEST_CHECK(test, limiter.shouldOutput(timing));

    timing.hasSensorSofTimestampTsc = false;
    timing.hasSensorTimestamp = true;
    timing.sensorTimestampNs = 2000000000ULL;
    TEST_CHECK(test, limiter.shouldOutput(timing));

    TEST_CHECK_EQ(test, argus_sender::signedDeltaNs(20, 10), 10LL);
    TEST_CHECK_EQ(test, argus_sender::signedDeltaNs(10, 20), -10LL);

    return test.finish();
}
