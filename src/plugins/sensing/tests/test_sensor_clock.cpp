// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "core/sensor_clock.hpp"

#include <catch2/catch_test_macros.hpp>

using plugins::sensing::ClockPair;
using plugins::sensing::SensorClockMap;
using plugins::sensing::SensorTimeDomain;

namespace
{
constexpr int64_t kHz = 31'250'000; // 32 ns per tick
constexpr int64_t kMs = 1'000'000;

SensorClockMap counter_map(int& reads, int64_t tick0 = 1'000'000, int64_t mono0 = 50'000 * kMs)
{
    SensorClockMap::Config cfg;
    cfg.domain = SensorTimeDomain::GenericCounterTicks;
    cfg.counter_hz = kHz;
    return SensorClockMap(cfg,
                          [&reads, tick0, mono0]
                          {
                              ++reads;
                              return ClockPair{ tick0, mono0, 1000 };
                          });
}
} // namespace

TEST_CASE("counter ticks convert through the anchor at the counter rate")
{
    int reads = 0;
    auto map = counter_map(reads);
    // 100 ms after the anchor = 3,125,000 ticks.
    const int64_t ready = 50'200 * kMs;
    REQUIRE(map.to_monotonic_ns(1'000'000 + 3'125'000, ready, ready) == 50'100 * kMs);
}

TEST_CASE("a capture before the anchor converts with a negative delta")
{
    int reads = 0;
    auto map = counter_map(reads);
    const int64_t ready = 50'010 * kMs;
    // 3,125 ticks before the anchor = 100 us.
    REQUIRE(map.to_monotonic_ns(1'000'000 - 3'125, ready, ready) == 50'000 * kMs - 100'000);
}

TEST_CASE("raw nanoseconds convert one to one")
{
    SensorClockMap::Config cfg;
    cfg.domain = SensorTimeDomain::MonotonicRawNs;
    SensorClockMap map(cfg, [] { return ClockPair{ 7'000 * kMs, 7'002 * kMs, 100 }; });
    REQUIRE(map.to_monotonic_ns(7'010 * kMs, 7'050 * kMs, 7'050 * kMs) == 7'012 * kMs);
}

TEST_CASE("the anchor is refreshed at the configured rate, not per frame")
{
    int reads = 0;
    auto map = counter_map(reads);
    const int64_t t = 50'100 * kMs;
    for (int i = 0; i < 30; ++i)
        map.to_monotonic_ns(1'000'000 + 3'125'000, t, t + i * 10 * kMs);
    REQUIRE(reads == 1);
    map.to_monotonic_ns(1'000'000 + 3'125'000, 51'200 * kMs, 51'200 * kMs);
    REQUIRE(reads == 2);
}

TEST_CASE("a capture in the future of the ready time is rejected")
{
    int reads = 0;
    auto map = counter_map(reads);
    REQUIRE_THROWS_AS(map.to_monotonic_ns(1'000'000 + 3'125'000, 50'050 * kMs, 50'050 * kMs), std::runtime_error);
}

TEST_CASE("a capture implausibly old is rejected, as with the wrong rate")
{
    int reads = 0;
    auto map = counter_map(reads);
    // Treating nanoseconds as ticks puts the capture far from the anchor.
    REQUIRE_THROWS_AS(map.to_monotonic_ns(1'000'000 + 400'000'000, 50'100 * kMs, 50'100 * kMs), std::runtime_error);
    int more = 0;
    auto old_map = counter_map(more, 200'000'000);
    // 199,999,999 ticks before the anchor is about 6.4 s.
    REQUIRE_THROWS_AS(old_map.to_monotonic_ns(1, 50'100 * kMs, 50'100 * kMs), std::runtime_error);
}

TEST_CASE("a domain is required, and counter ticks need a rate")
{
    SensorClockMap::Config cfg;
    auto reader = [] { return ClockPair{}; };
    REQUIRE_THROWS_AS(SensorClockMap(cfg, reader), std::invalid_argument);
    cfg.domain = SensorTimeDomain::GenericCounterTicks;
    REQUIRE_THROWS_AS(SensorClockMap(cfg, reader), std::invalid_argument);
}
