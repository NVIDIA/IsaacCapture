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
        // The anchor is read at `now`; the event is 30 ms before it, and 30 ms * 500 ppm is 15000 ns.
        TEST_CHECK_EQ(test, mapping.uncertaintyNs, clocks.bracketNs + 15000);
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
        // The anchor was read at the refresh, 400 ms before this frame was handled; the frame's event was 30 ms before
        // it, so 370 ms after the anchor. The bracket plus 500 ppm of that 370 ms distance, not of the 400 ms age.
        TEST_CHECK_EQ(test, aged.uncertaintyNs, clocks.bracketNs + 185000);
    }

    // The slew bound follows the distance from the anchor to the sensor event, not the anchor's age.
    {
        // The first frame after an anchor refresh has age 0; its event 22 ms before the anchor still costs 22 ms of
        // slew.
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping first = map.map(static_cast<uint64_t>((now - 22 * kMs) / 32), now, now);
        TEST_CHECK(test, first.valid);
        TEST_CHECK_EQ(test, first.anchorAgeNs, static_cast<int64_t>(0));
        TEST_CHECK_EQ(test, first.uncertaintyNs, clocks.bracketNs + 11000);
    }
    {
        // A delayed capture, 500 ms before the anchor but within maxAgeNs: 500 ms * 500 ppm, with a fresh anchor.
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping delayed = map.map(static_cast<uint64_t>((now - 500 * kMs) / 32), now, now);
        TEST_CHECK(test, delayed.valid);
        TEST_CHECK_EQ(test, delayed.anchorAgeNs, static_cast<int64_t>(0));
        TEST_CHECK_EQ(test, delayed.uncertaintyNs, clocks.bracketNs + 250000);
    }
    {
        // An event after the anchor: the anchor is read at T, the frame handled 300 ms later with its event 30 ms
        // before that, so 270 ms after the anchor. The age (300 ms) is reported separately from the uncertainty.
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        TEST_CHECK(test, map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now).valid);
        now += 300 * kMs;
        const SensorTimeMapping later = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, later.valid);
        TEST_CHECK_EQ(test, later.anchorAgeNs, 300 * kMs);
        TEST_CHECK_EQ(test, later.uncertaintyNs, clocks.bracketNs + 135000);
    }
    {
        // A non-default slewPpm is honoured: 300 ms before the anchor at 100 ppm.
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap::Config config;
        config.counterHz = kHz;
        config.slewPpm = 100;
        SensorClockMap map(config,
                           [&clocks, &now](ClockPair* pair)
                           {
                               *pair = clocks.at(now);
                               return true;
                           });
        const SensorTimeMapping mapping = map.map(static_cast<uint64_t>((now - 300 * kMs) / 32), now, now);
        TEST_CHECK(test, mapping.valid);
        TEST_CHECK_EQ(test, mapping.uncertaintyNs, clocks.bracketNs + 30000);
    }
    {
        // An event exactly at the anchor adds nothing: the uncertainty is the bracket, never less.
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping atAnchor = map.map(static_cast<uint64_t>(now / 32), now, now);
        TEST_CHECK(test, atAnchor.valid);
        TEST_CHECK_EQ(test, atAnchor.uncertaintyNs, clocks.bracketNs);
    }
    {
        // A refused conversion keeps the bound over the distance it computed (here 5 ms after the anchor, 5 s before
        // it); one refused before any conversion (no frequency, unreadable anchor, clock step) reports 0.
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping after = map.map(static_cast<uint64_t>((now + 5 * kMs) / 32), now, now);
        TEST_CHECK(test, !after.valid);
        TEST_CHECK_EQ(test, after.uncertaintyNs, clocks.bracketNs + 2500);
        const SensorTimeMapping tooOld = map.map(static_cast<uint64_t>((now - 5000 * kMs) / 32), now, now);
        TEST_CHECK(test, !tooOld.valid);
        TEST_CHECK_EQ(test, tooOld.uncertaintyNs, clocks.bracketNs + 2500000);

        FakeClocks stepClocks;
        int64_t stepNow = 10000 * kMs;
        SensorClockMap stepMap = makeMap(&stepClocks, &stepNow);
        stepMap.map(static_cast<uint64_t>((stepNow - 30 * kMs) / 32), stepNow, stepNow);
        stepNow += 1000 * kMs;
        stepClocks.skewNs = 50 * kMs;
        const SensorTimeMapping stepped = stepMap.map(static_cast<uint64_t>((stepNow - 30 * kMs) / 32), stepNow, stepNow);
        TEST_CHECK(test, !stepped.valid);
        TEST_CHECK_EQ(test, stepped.uncertaintyNs, static_cast<int64_t>(0));

        SensorClockMap::Config unknown;
        SensorClockMap noHz(unknown, [](ClockPair*) { return true; });
        TEST_CHECK_EQ(test, noHz.map(1, now, now).uncertaintyNs, static_cast<int64_t>(0));
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
