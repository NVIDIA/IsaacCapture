// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <remote_plugin_session/clock_offset_estimator.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using remote_plugin_session::ClockOffsetEstimator;
using remote_plugin_session::PingScheduler;
using remote_plugin_session::should_drop_stale;

TEST_CASE("ClockOffsetEstimator starts empty and follows the sign convention", "[remote_device_bridge][unit]")
{
    ClockOffsetEstimator estimator;
    REQUIRE(estimator.offset_ns() == 0);
    REQUIRE(estimator.remote_to_local(42) == 42);

    // Remote clock reads 1'000 when the local clock reads 5'000 (+ 100 of delay).
    estimator.add_sample(1'000, 5'100);
    REQUIRE(estimator.offset_ns() == 4'100);
    REQUIRE(estimator.remote_to_local(1'000) == 5'100);
}

TEST_CASE("ClockOffsetEstimator keeps the minimum delay sample", "[remote_device_bridge][unit]")
{
    ClockOffsetEstimator estimator;
    estimator.add_sample(1'000, 5'300); // delay 300
    estimator.add_sample(2'000, 6'050); // delay 50
    estimator.add_sample(3'000, 7'400); // delay 400
    REQUIRE(estimator.offset_ns() == 4'050);
}

TEST_CASE("ClockOffsetEstimator evicts by sample count", "[remote_device_bridge][unit]")
{
    ClockOffsetEstimator estimator(/*max_samples=*/2, /*window_ns=*/1'000'000);
    estimator.add_sample(0, 10); // offset 10 (the eventual minimum, evicted first)
    estimator.add_sample(100, 150); // offset 50
    estimator.add_sample(200, 260); // offset 60
    REQUIRE(estimator.offset_ns() == 50);
}

TEST_CASE("ClockOffsetEstimator evicts by time window", "[remote_device_bridge][unit]")
{
    ClockOffsetEstimator estimator(/*max_samples=*/100, /*window_ns=*/1'000);
    estimator.add_sample(0, 10); // local 10, offset 10
    estimator.add_sample(500, 800); // local 800, offset 300
    REQUIRE(estimator.offset_ns() == 10);
    estimator.add_sample(1'500, 1'700); // local 1'700, offset 200 -> drops samples older than 700
    REQUIRE(estimator.offset_ns() == 200);
}

TEST_CASE("ClockOffsetEstimator handles negative offsets", "[remote_device_bridge][unit]")
{
    ClockOffsetEstimator estimator;
    estimator.add_sample(10'000, 2'000);
    REQUIRE(estimator.offset_ns() == -8'000);
    REQUIRE(estimator.remote_to_local(10'500) == 2'500);
}

TEST_CASE("ClockOffsetEstimator rejects an empty window", "[remote_device_bridge][unit]")
{
    REQUIRE_THROWS_AS(ClockOffsetEstimator(0, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(ClockOffsetEstimator(1, 0), std::invalid_argument);
}

TEST_CASE("should_drop_stale keeps fresh samples and drops the backlog after a stall", "[remote_device_bridge][unit]")
{
    constexpr int64_t kMaxAge = 250'000'000; // 250 ms
    const int64_t now = 10'000'000'000;
    REQUIRE_FALSE(should_drop_stale(now, now, kMaxAge));
    REQUIRE_FALSE(should_drop_stale(now - kMaxAge, now, kMaxAge)); // exactly max_age is still fresh
    REQUIRE_FALSE(should_drop_stale(now + 5'000'000, now, kMaxAge)); // clock-offset bias may map into the future
    REQUIRE(should_drop_stale(now - kMaxAge - 1, now, kMaxAge));
    REQUIRE(should_drop_stale(now - 3'000'000'000, now, kMaxAge));
    REQUIRE_FALSE(should_drop_stale(now - 3'000'000'000, now, 0)); // 0 disables the guard
    REQUIRE_FALSE(should_drop_stale(now - 3'000'000'000, now, -1));
}

namespace
{

constexpr int64_t kMs = 1'000'000;

//! One exchange with true offset `offset` (local - remote), forward delay d_fwd, return delay d_back.
void exchange(ClockOffsetEstimator& estimator, int64_t t1, int64_t offset, int64_t d_fwd, int64_t d_back)
{
    const int64_t t2 = t1 + d_fwd - offset;
    const int64_t t3 = t2 + 1 * kMs; // client turnaround, excluded from the RTT
    const int64_t t4 = t3 + offset + d_back;
    estimator.add_two_way_sample(t1, t2, t3, t4);
}

} // namespace

TEST_CASE("Two-way exchanges recover the offset the one-way minimum biases", "[remote_device_bridge][unit]")
{
    constexpr int64_t kOffset = 5'000 * kMs;
    ClockOffsetEstimator estimator;
    // One-way: the fastest frame took 70 ms, so the one-way estimate is 70 ms too large.
    estimator.add_sample(1'000 * kMs, 1'000 * kMs + kOffset + 70 * kMs);
    REQUIRE(estimator.offset_ns() == kOffset + 70 * kMs);
    REQUIRE(std::string(estimator.offset_source()) == "one-way");
    REQUIRE(estimator.rtt_min_ns() == 0);

    exchange(estimator, 2'000 * kMs, kOffset, 70 * kMs, 70 * kMs);
    REQUIRE(std::string(estimator.offset_source()) == "two-way");
    REQUIRE(estimator.offset_ns() == kOffset);
    REQUIRE(estimator.rtt_last_ns() == 140 * kMs);
    REQUIRE(estimator.rtt_min_ns() == 140 * kMs);
    REQUIRE(estimator.remote_to_local(0) == kOffset);

    // One-way samples no longer move a two-way estimate.
    estimator.add_sample(3'000 * kMs, 3'000 * kMs + kOffset + 1 * kMs);
    REQUIRE(estimator.offset_ns() == kOffset);
}

TEST_CASE("Two-way offset error is half the path asymmetry", "[remote_device_bridge][unit]")
{
    constexpr int64_t kOffset = -123'456 * kMs;
    ClockOffsetEstimator estimator;
    // 10 ms out, 50 ms back: RTT 60 ms, estimate off by (50 - 10) / 2 = +20 ms.
    exchange(estimator, 1'000 * kMs, kOffset, 10 * kMs, 50 * kMs);
    REQUIRE(estimator.rtt_last_ns() == 60 * kMs);
    REQUIRE(estimator.offset_ns() - kOffset == 20 * kMs);
}

TEST_CASE("Two-way estimate follows the minimum-RTT exchange of the last 32", "[remote_device_bridge][unit]")
{
    constexpr int64_t kOffset = 7 * kMs;
    ClockOffsetEstimator estimator;
    // Exchange 0 has the lowest RTT and an asymmetry of +4 ms; the rest are slower and symmetric-ish.
    exchange(estimator, 0, kOffset, 1 * kMs, 9 * kMs); // rtt 10, error +4
    for (int i = 1; i < 32; ++i)
    {
        exchange(estimator, i * 1'000 * kMs, kOffset, 30 * kMs, 50 * kMs); // rtt 80, error +10
    }
    REQUIRE(estimator.rtt_min_ns() == 10 * kMs);
    REQUIRE(estimator.offset_ns() == kOffset + 4 * kMs);
    REQUIRE(estimator.rtt_last_ns() == 80 * kMs);

    exchange(estimator, 32'000 * kMs, kOffset, 30 * kMs, 50 * kMs); // evicts exchange 0
    REQUIRE(estimator.rtt_min_ns() == 80 * kMs);
    REQUIRE(estimator.offset_ns() == kOffset + 10 * kMs);

    estimator.add_two_way_sample(0, 100, 200, 50); // negative RTT: ignored
    REQUIRE(estimator.rtt_last_ns() == 80 * kMs);
}

TEST_CASE("PingScheduler bursts 10 PINGs at 50 ms, then 1 Hz", "[remote_device_bridge][unit]")
{
    constexpr int64_t kStart = 1'000 * kMs;
    PingScheduler pings(kStart);
    std::vector<int64_t> sent;
    for (int64_t t = kStart; t <= kStart + 3'500 * kMs; t += kMs)
    {
        const PingScheduler::Action action = pings.poll(t);
        REQUIRE(action != PingScheduler::Action::GiveUp);
        if (action == PingScheduler::Action::Ping)
        {
            sent.push_back(t - kStart);
            pings.on_pong();
        }
    }
    const std::vector<int64_t> expected = { 0,          50 * kMs,   100 * kMs, 150 * kMs, 200 * kMs,
                                            250 * kMs,  300 * kMs,  350 * kMs, 400 * kMs, 450 * kMs,
                                            1450 * kMs, 2450 * kMs, 3450 * kMs };
    REQUIRE(sent == expected);
    REQUIRE(pings.active());
}

TEST_CASE("PingScheduler gives up once when no PONG arrives within 2 s", "[remote_device_bridge][unit]")
{
    PingScheduler pings(0);
    int given_up = 0;
    int sent_after = 0;
    for (int64_t t = 0; t <= 5'000 * kMs; t += kMs)
    {
        switch (pings.poll(t))
        {
        case PingScheduler::Action::GiveUp:
            REQUIRE(t == PingScheduler::kPongTimeoutNs);
            ++given_up;
            break;
        case PingScheduler::Action::Ping:
            sent_after += given_up;
            break;
        case PingScheduler::Action::None:
            break;
        }
    }
    REQUIRE(given_up == 1);
    REQUIRE(sent_after == 0);
    REQUIRE_FALSE(pings.active());
}
