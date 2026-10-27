// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <remote_link/clock_offset.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using core::ClockOffsetEstimator;
using core::PingScheduler;

namespace
{

constexpr int64_t kMs = 1'000'000;

//! One exchange with true offset `offset` (local - remote), forward delay d_fwd, return delay d_back.
void exchange(ClockOffsetEstimator& estimator, int64_t t1, int64_t offset, int64_t d_fwd, int64_t d_back)
{
    const int64_t t2 = t1 + d_fwd - offset;
    const int64_t t3 = t2 + 1 * kMs; // peer turnaround, excluded from the RTT
    const int64_t t4 = t3 + offset + d_back;
    estimator.add_two_way_sample(t1, t2, t3, t4);
}

} // namespace

TEST_CASE("RemoteLink: ClockOffsetEstimator starts empty and follows offset = local - remote", "[remote_link][unit]")
{
    ClockOffsetEstimator estimator;
    REQUIRE_FALSE(estimator.has_sample());
    REQUIRE(estimator.offset_ns() == 0);
    REQUIRE(estimator.remote_to_local(42) == 42);

    estimator.add_sample(1'000, 5'100);
    REQUIRE(estimator.has_sample());
    REQUIRE_FALSE(estimator.two_way());
    REQUIRE(estimator.offset_ns() == 4'100);
    REQUIRE(estimator.remote_to_local(1'000) == 5'100);
}

TEST_CASE("RemoteLink: ClockOffsetEstimator keeps the minimum-delay one-way sample", "[remote_link][unit]")
{
    ClockOffsetEstimator estimator;
    estimator.add_sample(1'000, 5'300);
    estimator.add_sample(2'000, 6'050);
    estimator.add_sample(3'000, 7'400);
    REQUIRE(estimator.offset_ns() == 4'050);
}

TEST_CASE("RemoteLink: ClockOffsetEstimator evicts by count and by time window", "[remote_link][unit]")
{
    ClockOffsetEstimator by_count(/*max_samples=*/2, /*window_ns=*/1'000'000);
    by_count.add_sample(0, 10);
    by_count.add_sample(100, 150);
    by_count.add_sample(200, 260);
    REQUIRE(by_count.offset_ns() == 50);

    ClockOffsetEstimator by_time(/*max_samples=*/100, /*window_ns=*/1'000);
    by_time.add_sample(0, 10);
    by_time.add_sample(500, 800);
    REQUIRE(by_time.offset_ns() == 10);
    by_time.add_sample(1'500, 1'700); // drops samples received before 700
    REQUIRE(by_time.offset_ns() == 200);
}

TEST_CASE("RemoteLink: ClockOffsetEstimator handles negative offsets and rejects empty windows", "[remote_link][unit]")
{
    ClockOffsetEstimator estimator;
    estimator.add_sample(10'000, 2'000);
    REQUIRE(estimator.offset_ns() == -8'000);
    REQUIRE(estimator.remote_to_local(10'500) == 2'500);
    REQUIRE_THROWS_AS(ClockOffsetEstimator(0, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(ClockOffsetEstimator(1, 0), std::invalid_argument);
}

TEST_CASE("RemoteLink: two-way exchanges replace the biased one-way estimate", "[remote_link][unit]")
{
    constexpr int64_t kOffset = 5'000 * kMs;
    ClockOffsetEstimator estimator;
    estimator.add_sample(1'000 * kMs, 1'000 * kMs + kOffset + 70 * kMs);
    REQUIRE(estimator.offset_ns() == kOffset + 70 * kMs);
    REQUIRE(estimator.rtt_min_ns() == 0);

    exchange(estimator, 2'000 * kMs, kOffset, 70 * kMs, 70 * kMs);
    REQUIRE(estimator.two_way());
    REQUIRE(estimator.offset_ns() == kOffset);
    REQUIRE(estimator.rtt_last_ns() == 140 * kMs);
    REQUIRE(estimator.rtt_min_ns() == 140 * kMs);
    REQUIRE(estimator.remote_to_local(0) == kOffset);

    estimator.add_sample(3'000 * kMs, 3'000 * kMs + kOffset + 1 * kMs);
    REQUIRE(estimator.offset_ns() == kOffset);
}

TEST_CASE("RemoteLink: two-way offset error is half the path asymmetry", "[remote_link][unit]")
{
    constexpr int64_t kOffset = -123'456 * kMs;
    ClockOffsetEstimator estimator;
    exchange(estimator, 1'000 * kMs, kOffset, 10 * kMs, 50 * kMs);
    REQUIRE(estimator.rtt_last_ns() == 60 * kMs);
    REQUIRE(estimator.offset_ns() - kOffset == 20 * kMs);
}

TEST_CASE("RemoteLink: two-way estimate follows the minimum-RTT exchange of the last 32", "[remote_link][unit]")
{
    constexpr int64_t kOffset = 7 * kMs;
    ClockOffsetEstimator estimator;
    exchange(estimator, 0, kOffset, 1 * kMs, 9 * kMs); // rtt 10, error +4
    for (int i = 1; i < 32; ++i)
    {
        exchange(estimator, i * 1'000 * kMs, kOffset, 30 * kMs, 50 * kMs); // rtt 80, error +10
    }
    REQUIRE(estimator.rtt_min_ns() == 10 * kMs);
    REQUIRE(estimator.offset_ns() == kOffset + 4 * kMs);
    REQUIRE(estimator.rtt_last_ns() == 80 * kMs);

    exchange(estimator, 32'000 * kMs, kOffset, 30 * kMs, 50 * kMs); // evicts the first exchange
    REQUIRE(estimator.rtt_min_ns() == 80 * kMs);
    REQUIRE(estimator.offset_ns() == kOffset + 10 * kMs);

    estimator.add_two_way_sample(0, 100, 200, 50); // negative RTT: ignored
    REQUIRE(estimator.rtt_last_ns() == 80 * kMs);
}

TEST_CASE("RemoteLink: PingScheduler bursts 10 PINGs at 50 ms, then 1 Hz", "[remote_link][unit]")
{
    constexpr int64_t kStart = 1'000 * kMs;
    PingScheduler pings(kStart);
    REQUIRE(pings.next_due_ns() == kStart);
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
    REQUIRE(pings.next_due_ns() == kStart + 4'450 * kMs);
}

TEST_CASE("RemoteLink: PingScheduler gives up once without a PONG within 2 s", "[remote_link][unit]")
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
            if (t >= 1'450 * kMs)
            {
                // The give-up deadline caps the next due time while no PONG arrived.
                REQUIRE(pings.next_due_ns() == PingScheduler::kPongTimeoutNs);
            }
            break;
        case PingScheduler::Action::None:
            break;
        }
    }
    REQUIRE(given_up == 1);
    REQUIRE(sent_after == 0);
    REQUIRE_FALSE(pings.active());
    REQUIRE(pings.next_due_ns() == std::numeric_limits<int64_t>::max());
}
