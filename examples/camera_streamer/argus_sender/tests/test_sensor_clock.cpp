/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SensorClock.h"
#include "TestHarness.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

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
    // Added to CLOCK_MONOTONIC for the realtime reading; a wall-clock step changes it. Off unless hasRealtime.
    int64_t realtimeOffsetNs = 0;
    bool hasRealtime = false;
    int reads = 0;

    ClockPair at(int64_t monotonicNs)
    {
        ++reads;
        ClockPair pair;
        pair.source = (monotonicNs + skewNs) / 32;
        pair.monotonicNs = monotonicNs;
        pair.bracketNs = bracketNs;
        pair.hasRealtime = hasRealtime;
        if (hasRealtime)
            pair.realtimeNs = monotonicNs + realtimeOffsetNs;
        return pair;
    }
};

SensorClockMap makeMap(FakeClocks* clocks, int64_t* nowNs)
{
    SensorClockMap::Config config;
    config.counterHz = kHz;
    // The cases below are written around a one-second refresh and a 500 ppm slew; the defaults are covered separately.
    config.refreshNs = 1000000000;
    config.slewPpm = 500;
    return SensorClockMap(config,
                          [clocks, nowNs](ClockPair* pair)
                          {
                              *pair = clocks->at(*nowNs);
                              return true;
                          });
}

// A 1 GHz counter (one tick per ns) whose pair reader returns whatever `anchor` holds when it is called.
SensorClockMap::Config gigahertzConfig()
{
    SensorClockMap::Config config;
    config.counterHz = 1000000000;
    config.refreshNs = 1000000000;
    return config;
}

SensorClockMap makeAnchorMap(const SensorClockMap::Config& config, ClockPair* anchor)
{
    return SensorClockMap(config,
                          [anchor](ClockPair* pair)
                          {
                              *pair = *anchor;
                              return true;
                          });
}

// A counter of `hz` Hz that read 0 when CLOCK_MONOTONIC read 0, plus a realtime clock `realtimeOffsetNs` ahead. The
// reader returns raw ticks, as readCounterPair does.
struct TickClock
{
    int64_t hz = 0;
    int64_t* nowNs = nullptr;
    int64_t realtimeOffsetNs = 1700000000000000000LL;
    int reads = 0;
    bool fail = false;

    int64_t ticksAt(int64_t monotonicNs) const
    {
        return static_cast<int64_t>(static_cast<__int128>(monotonicNs) * hz / 1000000000LL);
    }

    SensorClockMap::PairReader reader()
    {
        return [this](ClockPair* pair)
        {
            ++reads;
            if (fail)
                return false;
            pair->source = ticksAt(*nowNs);
            pair->monotonicNs = *nowNs;
            pair->bracketNs = 2000;
            pair->hasRealtime = true;
            pair->realtimeNs = *nowNs + realtimeOffsetNs;
            return true;
        };
    }
};

SensorClockMap::Config tickConfig()
{
    SensorClockMap::Config config;
    config.refreshNs = 1000000000;
    return config;
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

    // The realtime capture time is the monotonic one plus the anchor's realtime-minus-monotonic offset.
    {
        FakeClocks clocks;
        clocks.hasRealtime = true;
        clocks.realtimeOffsetNs = 1700000000LL * 1000000000LL + 123456;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping mapping = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, mapping.valid);
        TEST_CHECK(test, mapping.hasRealtime);
        TEST_CHECK_EQ(test, mapping.captureMonotonicNs, now - 30 * kMs);
        TEST_CHECK_EQ(test, mapping.captureRealtimeNs, mapping.captureMonotonicNs + clocks.realtimeOffsetNs);
    }

    // A reader that gave no realtime reading leaves the mapping without one; the monotonic result is unchanged.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        const SensorTimeMapping mapping = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, mapping.valid);
        TEST_CHECK(test, !mapping.hasRealtime);
        TEST_CHECK_EQ(test, mapping.captureRealtimeNs, static_cast<int64_t>(0));
        TEST_CHECK_EQ(test, mapping.captureMonotonicNs, now - 30 * kMs);
    }

    // A wall-clock step between anchors, larger than stepThresholdNs plus both brackets, invalidates the mapping
    // although the counter still agrees with CLOCK_MONOTONIC; the next consistent refresh recovers.
    {
        FakeClocks clocks;
        clocks.hasRealtime = true;
        clocks.realtimeOffsetNs = 1700000000LL * 1000000000LL;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        TEST_CHECK(test, map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now).valid);
        now += 1000 * kMs;
        clocks.realtimeOffsetNs += 50 * kMs;
        const SensorTimeMapping stepped = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, !stepped.valid);
        TEST_CHECK(test, stepped.reason.find("clock step") != std::string::npos);
        now += 1000 * kMs;
        const SensorTimeMapping recovered = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, recovered.valid);
        TEST_CHECK_EQ(test, recovered.captureRealtimeNs, recovered.captureMonotonicNs + clocks.realtimeOffsetNs);
    }

    // A realtime offset movement inside the allowance (2 ms threshold plus two 2 us brackets) is not a step.
    {
        FakeClocks clocks;
        clocks.hasRealtime = true;
        clocks.realtimeOffsetNs = 1700000000LL * 1000000000LL;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        TEST_CHECK(test, map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now).valid);
        now += 1000 * kMs;
        clocks.realtimeOffsetNs += 1 * kMs;
        const SensorTimeMapping within = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, within.valid);
        TEST_CHECK_EQ(test, within.captureRealtimeNs, within.captureMonotonicNs + clocks.realtimeOffsetNs);
    }

    // A counter step is still caught when the realtime offset holds still, and a realtime step on an anchor pair
    // where one side has no realtime reading is not judged.
    {
        FakeClocks clocks;
        clocks.hasRealtime = true;
        int64_t now = 10000 * kMs;
        SensorClockMap map = makeMap(&clocks, &now);
        TEST_CHECK(test, map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now).valid);
        now += 1000 * kMs;
        clocks.skewNs = 50 * kMs;
        const SensorTimeMapping stepped = map.map(static_cast<uint64_t>((now - 30 * kMs) / 32), now, now);
        TEST_CHECK(test, !stepped.valid);
        TEST_CHECK(test, stepped.reason.find("clock step") != std::string::npos);

        FakeClocks mixed;
        int64_t mixedNow = 10000 * kMs;
        SensorClockMap mixedMap = makeMap(&mixed, &mixedNow);
        TEST_CHECK(test, mixedMap.map(static_cast<uint64_t>((mixedNow - 30 * kMs) / 32), mixedNow, mixedNow).valid);
        mixedNow += 1000 * kMs;
        mixed.hasRealtime = true;
        mixed.realtimeOffsetNs = 5000 * kMs;
        const SensorTimeMapping unjudged =
            mixedMap.map(static_cast<uint64_t>((mixedNow - 30 * kMs) / 32), mixedNow, mixedNow);
        TEST_CHECK(test, unjudged.valid);
        TEST_CHECK(test, unjudged.hasRealtime);
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

    // The monotonic clock slewing against the counter between anchors: with the anchor read at 1 s and the event 0.5 s
    // later, +10000 ppm puts the true monotonic time at 1.505 s. A 500 ppm bound (252000 ns) did not cover that 5 ms.
    {
        const int64_t anchorNs = 1000 * kMs;
        const int64_t trueNs = 1505 * kMs;
        ClockPair anchor;
        anchor.source = anchorNs;
        anchor.monotonicNs = anchorNs;
        anchor.bracketNs = 2000;

        SensorClockMap::Config wide = gigahertzConfig();
        wide.maxUncertaintyNs = 1000 * kMs;
        SensorClockMap map = makeAnchorMap(wide, &anchor);
        TEST_CHECK(test, map.map(static_cast<uint64_t>(anchorNs), anchorNs, anchorNs).valid);
        const SensorTimeMapping mapping = map.map(static_cast<uint64_t>(1500 * kMs), trueNs, trueNs);
        TEST_CHECK(test, mapping.valid);
        TEST_CHECK(test, std::llabs(mapping.captureMonotonicNs - trueNs) <= mapping.uncertaintyNs);

        // The default ceiling refuses a bound this wide rather than reporting it.
        SensorClockMap defaulted = makeAnchorMap(gigahertzConfig(), &anchor);
        TEST_CHECK(test, defaulted.map(static_cast<uint64_t>(anchorNs), anchorNs, anchorNs).valid);
        const SensorTimeMapping refused = defaulted.map(static_cast<uint64_t>(1500 * kMs), trueNs, trueNs);
        TEST_CHECK(test, !refused.valid);
        TEST_CHECK(test, refused.reason.find("too far from the anchor") != std::string::npos);
    }

    // Between refreshes, a slew up to the configured rate never moves the true time outside the reported uncertainty.
    {
        const int64_t anchorNs = 1000 * kMs;
        const int64_t slews[] = { -500, 500, -10000, 10000, -83000, 83000 };
        const int64_t distances[] = { 20 * kMs, 100 * kMs, 400 * kMs };
        for (const int64_t slew : slews)
        {
            for (const int64_t distance : distances)
            {
                ClockPair anchor;
                anchor.source = anchorNs;
                anchor.monotonicNs = anchorNs;
                anchor.bracketNs = 2000;
                SensorClockMap::Config config = gigahertzConfig();
                config.refreshNs = 10000 * kMs;
                config.maxUncertaintyNs = 1000 * kMs; // 83000 ppm over 400 ms is past the default ceiling
                SensorClockMap map = makeAnchorMap(config, &anchor);
                TEST_CHECK(test, map.map(static_cast<uint64_t>(anchorNs), anchorNs, anchorNs).valid);
                // The monotonic clock runs `slew` ppm fast against the counter over the distance.
                const int64_t trueNs = anchorNs + distance + distance * slew / 1000000;
                const int64_t readyNs = std::max(trueNs, anchorNs + distance);
                const SensorTimeMapping mapping =
                    map.map(static_cast<uint64_t>(anchorNs + distance), readyNs, anchorNs + distance);
                TEST_CHECK(test, mapping.valid);
                if (mapping.valid)
                    TEST_CHECK(test, std::llabs(mapping.captureMonotonicNs - trueNs) <= mapping.uncertaintyNs);
            }
        }
    }

    // A consistent slew at the maximum rate between anchors is not a step; a 2 s jump of CLOCK_MONOTONIC is.
    {
        const int64_t anchorNs = 1000 * kMs;
        ClockPair anchor;
        anchor.source = anchorNs;
        anchor.monotonicNs = anchorNs;
        anchor.bracketNs = 2000;
        SensorClockMap::Config config = gigahertzConfig();
        config.refreshNs = 100 * kMs;
        SensorClockMap map = makeAnchorMap(config, &anchor);
        TEST_CHECK(test, map.map(static_cast<uint64_t>(anchorNs), anchorNs, anchorNs).valid);
        // 100 ms of counter time is 108.3 ms of CLOCK_MONOTONIC at +83000 ppm.
        anchor.monotonicNs = anchorNs + 100 * kMs + 8300000;
        anchor.source = anchorNs + 100 * kMs;
        const SensorTimeMapping slewed =
            map.map(static_cast<uint64_t>(anchor.source), anchor.monotonicNs, anchor.monotonicNs);
        TEST_CHECK(test, slewed.valid);

        ClockPair jumped = anchor;
        SensorClockMap jumpMap = makeAnchorMap(config, &jumped);
        jumped.monotonicNs = anchorNs;
        jumped.source = anchorNs;
        TEST_CHECK(test, jumpMap.map(static_cast<uint64_t>(anchorNs), anchorNs, anchorNs).valid);
        jumped.monotonicNs = anchorNs + 100 * kMs + 2000 * kMs;
        jumped.source = anchorNs + 100 * kMs;
        const SensorTimeMapping stepped =
            jumpMap.map(static_cast<uint64_t>(jumped.source), jumped.monotonicNs, jumped.monotonicNs);
        TEST_CHECK(test, !stepped.valid);
        TEST_CHECK(test, stepped.reason.find("clock step") != std::string::npos);
    }

    // The default maxUncertaintyNs: an event 241 ms from the anchor exceeds 20 ms of uncertainty, one 239 ms from it
    // does not.
    {
        FakeClocks clocks;
        int64_t now = 10000 * kMs;
        SensorClockMap::Config config;
        config.counterHz = kHz;
        SensorClockMap map(config,
                           [&clocks, &now](ClockPair* pair)
                           {
                               *pair = clocks.at(now);
                               return true;
                           });
        const SensorTimeMapping inside = map.map(static_cast<uint64_t>((now - 239 * kMs) / 32), now, now);
        TEST_CHECK(test, inside.valid);
        TEST_CHECK(test, inside.uncertaintyNs <= 20 * kMs);
        const SensorTimeMapping outside = map.map(static_cast<uint64_t>((now - 241 * kMs) / 32), now, now);
        TEST_CHECK(test, !outside.valid);
        TEST_CHECK(test, outside.uncertaintyNs > 20 * kMs);
        TEST_CHECK(test, outside.reason.find("too far from the anchor") != std::string::npos);
    }

    // The shared mapper, at counter rates that are not hardcoded anywhere.
    for (const int64_t hz : { 31250000LL, 19200000LL })
    {
        int64_t now = 10000 * kMs;
        TickClock clock;
        clock.hz = hz;
        clock.nowNs = &now;
        const int64_t ticks = clock.ticksAt(now - 20 * kMs);
        const int64_t ns = counterTicksToNs(ticks, hz);

        // The same event as ticks and as nanoseconds places identically.
        SensorTimeMapper fromTicks(hz, tickConfig(), clock.reader());
        SensorTimeMapper fromNs(hz, tickConfig(), clock.reader());
        const SensorTimeMapping a =
            fromTicks.map(static_cast<uint64_t>(ticks), SensorTimestampUnit::CounterTicks, now, now);
        const SensorTimeMapping b = fromNs.map(static_cast<uint64_t>(ns), SensorTimestampUnit::Nanoseconds, now, now);
        TEST_CHECK(test, a.valid);
        TEST_CHECK(test, b.valid);
        TEST_CHECK_EQ(test, a.captureMonotonicNs, b.captureMonotonicNs);
        TEST_CHECK_EQ(test, a.captureRealtimeNs, b.captureRealtimeNs);
        TEST_CHECK_EQ(test, a.hasRealtime, b.hasRealtime);
        TEST_CHECK_EQ(test, a.uncertaintyNs, b.uncertaintyNs);
        TEST_CHECK(test, std::llabs(a.captureMonotonicNs - (now - 20 * kMs)) <= a.uncertaintyNs);
        TEST_CHECK_EQ(test, a.captureRealtimeNs - a.captureMonotonicNs, clock.realtimeOffsetNs);
        // One counter tick of uncertainty is included, on top of the anchor bracket.
        TEST_CHECK(test, a.uncertaintyNs >= 2000 + (1000000000LL + hz - 1) / hz);

        // An unknown frequency refuses every mapping with a reason and no value.
        SensorTimeMapper noHz(0, tickConfig(), clock.reader());
        for (const SensorTimestampUnit unit : { SensorTimestampUnit::CounterTicks, SensorTimestampUnit::Nanoseconds })
        {
            const SensorTimeMapping refused = noHz.map(static_cast<uint64_t>(ticks), unit, now, now);
            TEST_CHECK(test, !refused.valid);
            TEST_CHECK(test, !refused.reason.empty());
            TEST_CHECK_EQ(test, refused.captureMonotonicNs, static_cast<int64_t>(0));
        }
        SensorTimeMapper negHz(-5, tickConfig(), clock.reader());
        TEST_CHECK(test, !negHz.map(static_cast<uint64_t>(ticks), SensorTimestampUnit::CounterTicks, now, now).valid);

        // A timestamp above INT64_MAX is out of range in either unit.
        SensorTimeMapper huge(hz, tickConfig(), clock.reader());
        const uint64_t tooBig = static_cast<uint64_t>(INT64_MAX) + 1;
        for (const SensorTimestampUnit unit : { SensorTimestampUnit::CounterTicks, SensorTimestampUnit::Nanoseconds })
        {
            const SensorTimeMapping refused = huge.map(tooBig, unit, now, now);
            TEST_CHECK(test, !refused.valid);
            TEST_CHECK(test, !refused.reason.empty());
        }
        TEST_CHECK(test, !huge.map(UINT64_MAX, SensorTimestampUnit::CounterTicks, now, now).valid);

        // A reader failure is invalid with a reason.
        TickClock broken = clock;
        broken.fail = true;
        SensorTimeMapper failing(hz, tickConfig(), broken.reader());
        const SensorTimeMapping failed =
            failing.map(static_cast<uint64_t>(ticks), SensorTimestampUnit::CounterTicks, now, now);
        TEST_CHECK(test, !failed.valid);
        TEST_CHECK(test, !failed.reason.empty());

        // applyMappingToTiming: all five capture fields for a valid mapping, none for a later invalid one.
        FrameTiming timing;
        applyMappingToTiming(a, &timing);
        TEST_CHECK(test, timing.hasCaptureMonotonic);
        TEST_CHECK_EQ(test, timing.captureMonotonicNs, a.captureMonotonicNs);
        TEST_CHECK_EQ(test, timing.captureUncertaintyNs, a.uncertaintyNs);
        TEST_CHECK_EQ(test, timing.captureAnchorAgeNs, a.anchorAgeNs);
        TEST_CHECK_EQ(test, timing.captureRealtimeNs, a.captureRealtimeNs);
        TEST_CHECK(test, timing.captureRealtimeNs != 0);
        applyMappingToTiming(failed, &timing);
        TEST_CHECK(test, !timing.hasCaptureMonotonic);
        TEST_CHECK_EQ(test, timing.captureMonotonicNs, static_cast<int64_t>(0));
        TEST_CHECK_EQ(test, timing.captureUncertaintyNs, static_cast<int64_t>(0));
        TEST_CHECK_EQ(test, timing.captureAnchorAgeNs, static_cast<int64_t>(0));
        TEST_CHECK_EQ(test, timing.captureRealtimeNs, static_cast<int64_t>(0));
    }

    // Two cameras are independent: each keeps its own anchor, and reset() forces a fresh read.
    {
        const int64_t hz = 19200000;
        int64_t nowA = 10000 * kMs;
        int64_t nowB = 50000 * kMs;
        TickClock clockA;
        clockA.hz = hz;
        clockA.nowNs = &nowA;
        TickClock clockB;
        clockB.hz = hz;
        clockB.nowNs = &nowB;
        clockB.realtimeOffsetNs = 1600000000000000000LL;
        SensorTimeMapper mapperA(hz, tickConfig(), clockA.reader());
        SensorTimeMapper mapperB(hz, tickConfig(), clockB.reader());
        const int64_t evA = nowA - 10 * kMs;
        const int64_t evB = nowB - 30 * kMs;
        const SensorTimeMapping a1 =
            mapperA.map(static_cast<uint64_t>(clockA.ticksAt(evA)), SensorTimestampUnit::CounterTicks, nowA, nowA);
        const SensorTimeMapping b1 =
            mapperB.map(static_cast<uint64_t>(clockB.ticksAt(evB)), SensorTimestampUnit::CounterTicks, nowB, nowB);
        TEST_CHECK(test, a1.valid);
        TEST_CHECK(test, b1.valid);
        TEST_CHECK(test, std::llabs(a1.captureMonotonicNs - evA) <= a1.uncertaintyNs);
        TEST_CHECK(test, std::llabs(b1.captureMonotonicNs - evB) <= b1.uncertaintyNs);
        TEST_CHECK_EQ(test, a1.captureRealtimeNs - a1.captureMonotonicNs, clockA.realtimeOffsetNs);
        TEST_CHECK_EQ(test, b1.captureRealtimeNs - b1.captureMonotonicNs, clockB.realtimeOffsetNs);
        TEST_CHECK_EQ(test, clockA.reads, 1);
        TEST_CHECK_EQ(test, clockB.reads, 1);

        // Within the refresh interval neither mapper reads again; a frame on A does not touch B's anchor.
        nowA += 100 * kMs;
        const int64_t evA2 = nowA - 5 * kMs;
        const SensorTimeMapping a2 =
            mapperA.map(static_cast<uint64_t>(clockA.ticksAt(evA2)), SensorTimestampUnit::CounterTicks, nowA, nowA);
        TEST_CHECK(test, a2.valid);
        TEST_CHECK_EQ(test, clockA.reads, 1);
        TEST_CHECK_EQ(test, clockB.reads, 1);

        // reset() on A forces a new anchor read on A only.
        mapperA.reset();
        const SensorTimeMapping a3 =
            mapperA.map(static_cast<uint64_t>(clockA.ticksAt(evA2)), SensorTimestampUnit::CounterTicks, nowA, nowA);
        TEST_CHECK(test, a3.valid);
        TEST_CHECK_EQ(test, clockA.reads, 2);
        TEST_CHECK_EQ(test, clockB.reads, 1);
        TEST_CHECK_EQ(test, a3.anchorAgeNs, static_cast<int64_t>(0));
    }

    TEST_CHECK(test, monotonicNowNs() > 0);

    return test.finish();
}
