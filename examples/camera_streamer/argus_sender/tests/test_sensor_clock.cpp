/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SensorClock.h"
#include "TestHarness.h"

using namespace argus_sender;

namespace
{

const int64_t kHz = 31250000; // 32 ns per tick
const int64_t kMs = 1000000;

// A counter that started at tick 0 when CLOCK_MONOTONIC read 0, with `skewNs` of disagreement added to its readings.
struct FakeClocks
{
    int64_t skewNs = 0;
    int64_t bracketNs = 2000;
    int reads = 0;

    ClockPair at(int64_t monotonicNs)
    {
        ++reads;
        ClockPair pair;
        pair.source = (monotonicNs + skewNs) / 32;
        pair.monotonicNs = monotonicNs;
        pair.bracketNs = bracketNs;
        return pair;
    }
};

SensorClockMap makeMap(FakeClocks* clocks, int64_t* nowNs)
{
    SensorClockMap::Config config;
    config.counterHz = kHz;
    return SensorClockMap(config,
                          [clocks, nowNs](ClockPair* pair)
                          {
                              *pair = clocks->at(*nowNs);
                              return true;
                          });
}

} // namespace

int main()
{
    TestContext test("sensor_clock");

    TEST_CHECK_EQ(test, counterTicksToNs(31250000, kHz), static_cast<int64_t>(1000000000));
    TEST_CHECK_EQ(test, counterTicksToNs(1, kHz), static_cast<int64_t>(32));
    TEST_CHECK_EQ(test, counterTicksToNs(-31250001, kHz), static_cast<int64_t>(-1000000032));

    // A frame captured 30 ms before it was ready maps to that instant.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const uint64_t ticks = static_cast<uint64_t>((now - 30 * kMs) / 32);
        const SensorTimeMapping mapping = map.map(ticks, now, now);
        TEST_CHECK(test, mapping.valid);
        TEST_CHECK_EQ(test, mapping.captureMonotonicNs, now - 30 * kMs);
        TEST_CHECK_EQ(test, mapping.uncertaintyNs, clocks.bracketNs);
    }

    // The sensor reporting nanoseconds where ticks were assumed lands far from the frame and is refused.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping mapping = map.map(static_cast<uint64_t>(now - 30 * kMs), now, now);
        TEST_CHECK(test, !mapping.valid);
        TEST_CHECK(test, !mapping.reason.empty());
    }

    // A capture after the frame was ready is refused, not clamped.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping mapping = map.map(static_cast<uint64_t>((now + 5 * kMs) / 32), now, now);
        TEST_CHECK(test, !mapping.valid);
        TEST_CHECK(test, mapping.reason.find("after the frame") != std::string::npos);
    }

    // A capture implausibly long before the frame was ready is refused.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping mapping = map.map(static_cast<uint64_t>((now - 5000 * kMs) / 32), now, now);
        TEST_CHECK(test, !mapping.valid);
    }

    // The anchor is read once and refreshed on a schedule, not per frame; its age is reported.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        for (int i = 0; i < 20; ++i)
        {
            now += 33 * kMs;
            map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        }
        TEST_CHECK_EQ(test, clocks.reads, 1);
        now += 1000 * kMs;
        const SensorTimeMapping refreshed = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, refreshed.valid);
        TEST_CHECK_EQ(test, clocks.reads, 2);
        TEST_CHECK_EQ(test, refreshed.anchorAgeNs, static_cast<int64_t>(0));
        now += 400 * kMs;
        const SensorTimeMapping aged = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK_EQ(test, aged.anchorAgeNs, 400 * kMs);
        // The bracket plus 500 ppm of the anchor's age.
        TEST_CHECK_EQ(test, aged.uncertaintyNs, clocks.bracketNs + 200000);
    }

    // A clock step between anchors invalidates the mapping until the next consistent refresh.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        TEST_CHECK(test, map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now).valid);
        now += 1000 * kMs;
        clocks.skewNs = 50 * kMs; // the counter jumped 50 ms relative to CLOCK_MONOTONIC
        const SensorTimeMapping stepped = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, !stepped.valid);
        TEST_CHECK(test, stepped.reason.find("clock step") != std::string::npos);
        now += 1000 * kMs;
        const SensorTimeMapping recovered =
            map.map(static_cast<uint64_t>((now - 30 * kMs + clocks.skewNs) / 32), now, now);
        TEST_CHECK(test, recovered.valid);
    }

    // Resetting (capture restart) drops the anchor: the next frame reads a fresh one and a step is forgotten.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        now += 100 * kMs;
        map.reset();
        TEST_CHECK(test, map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now).valid);
        TEST_CHECK_EQ(test, clocks.reads, 2);
    }

    // No counter on this platform, or no known frequency, is invalid with a reason, never a guess.
    {
        int64_t now = 10000 * kMs;
        SensorClockMap::Config config;
        config.counterHz = kHz;
        SensorClockMap unreadable(config, [](ClockPair*) { return false; });
        TEST_CHECK(test, !unreadable.map(1, now, now).valid);
        SensorClockMap::Config unknown;
        FakeClocks clocks;
        SensorClockMap noHz(unknown,
                            [&clocks, &now](ClockPair* pair)
                            {
                                *pair = clocks.at(now);
                                return true;
                            });
        TEST_CHECK(test, !noHz.map(1, now, now).valid);
    }

    return test.finish();
}
