// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <video_stage/frame_correlator.hpp>

#include <cstdint>
#include <random>
#include <vector>

using video_stage::FrameCorrelator;
using video_stage::InFlightFrame;

namespace
{

InFlightFrame frame(std::uint32_t sequence, std::int64_t submitted_us)
{
    InFlightFrame f;
    f.sequence = sequence;
    f.capture_tsc = 1000 + sequence;
    f.stamp_us = 5000 + sequence;
    f.submitted_us = submitted_us;
    return f;
}

} // namespace

TEST_CASE("stamp mode matches returned units in order", "[frame_correlator]")
{
    FrameCorrelator c(FrameCorrelator::Mode::Stamp);
    for (std::uint32_t i = 0; i < 5; ++i)
        c.submitted(frame(i, 100 + 33 * i));
    for (std::uint32_t i = 0; i < 5; ++i)
    {
        const auto r = c.returned(100 + 33 * i);
        REQUIRE(r.has_value());
        CHECK(r->identified);
        CHECK(r->skipped == 0);
        CHECK(r->frame.sequence == i);
        CHECK(r->frame.capture_tsc == 1000 + i);
    }
    CHECK(c.empty());
}

TEST_CASE("stamp mode counts a skipped input and keeps matching", "[frame_correlator]")
{
    FrameCorrelator c(FrameCorrelator::Mode::Stamp);
    c.submitted(frame(0, 10));
    c.submitted(frame(1, 20));
    c.submitted(frame(2, 30));
    auto a = c.returned(10);
    REQUIRE(a);
    CHECK(a->frame.sequence == 0);
    auto third = c.returned(30);
    REQUIRE(third);
    CHECK(third->identified);
    CHECK(third->frame.sequence == 2);
    CHECK(third->skipped == 1);
    CHECK(c.empty());

    c.submitted(frame(3, 40));
    auto later = c.returned(40);
    REQUIRE(later);
    CHECK(later->identified);
    CHECK(later->frame.sequence == 3);
    CHECK(later->skipped == 0);
}

TEST_CASE("stamp mode reports a unit that matches nothing as not identified", "[frame_correlator]")
{
    FrameCorrelator c(FrameCorrelator::Mode::Stamp);
    c.submitted(frame(0, 10));
    c.submitted(frame(1, 20));
    c.submitted(frame(2, 30));
    const auto r = c.returned(999);
    REQUIRE(r);
    CHECK_FALSE(r->identified);
    CHECK(r->skipped == 0);
    CHECK(r->frame.sequence == 0); // the oldest, removed and not credited
    CHECK(c.in_flight() == 2);

    const auto next = c.returned(20);
    REQUIRE(next);
    CHECK(next->identified);
    CHECK(next->frame.sequence == 1);
    CHECK(next->skipped == 0);
    const auto last = c.returned(30);
    REQUIRE(last);
    CHECK(last->identified);
    CHECK(last->frame.sequence == 2);
}

TEST_CASE("returned yields nothing when no frame is in flight", "[frame_correlator]")
{
    for (auto mode : { FrameCorrelator::Mode::Stamp, FrameCorrelator::Mode::Fifo })
    {
        FrameCorrelator c(mode);
        CHECK_FALSE(c.returned(1).has_value());
        c.submitted(frame(0, 1));
        REQUIRE(c.returned(1));
        CHECK_FALSE(c.returned(1).has_value());
    }
}

TEST_CASE("the in-flight queue is bounded by dropping the oldest", "[frame_correlator]")
{
    FrameCorrelator c(FrameCorrelator::Mode::Stamp, 12);
    for (std::uint32_t i = 0; i < 13; ++i)
        c.submitted(frame(i, 100 + i));
    CHECK(c.in_flight() == 12);

    // The dropped frame's unit arrives late: it matches nothing.
    const auto late = c.returned(100);
    REQUIRE(late);
    CHECK_FALSE(late->identified);
    CHECK(late->frame.sequence == 1); // the oldest remaining was removed, not credited

    const auto next = c.returned(102);
    REQUIRE(next);
    CHECK(next->identified);
    CHECK(next->frame.sequence == 2);
    CHECK(next->skipped == 0);
}

TEST_CASE("fifo mode takes the oldest regardless of stamp and never skips", "[frame_correlator]")
{
    FrameCorrelator c(FrameCorrelator::Mode::Fifo);
    c.submitted(frame(0, 10));
    c.submitted(frame(1, 20));
    c.submitted(frame(2, 30));
    for (std::uint32_t i = 0; i < 3; ++i)
    {
        const auto r = c.returned(30 - 10 * static_cast<std::int64_t>(i)); // stamps deliberately out of order
        REQUIRE(r);
        CHECK(r->identified);
        CHECK(r->skipped == 0);
        CHECK(r->frame.sequence == i);
    }
    CHECK(c.empty());
}

TEST_CASE("stamp mode: random submit and return sequences with skips", "[frame_correlator]")
{
    std::mt19937 rng(12345);
    for (int run = 0; run < 100; ++run)
    {
        FrameCorrelator c(FrameCorrelator::Mode::Stamp);
        std::vector<InFlightFrame> pending; // submitted, not yet returned or skipped
        std::uint32_t next = 0;
        std::int64_t stamp = 1000;
        for (int step = 0; step < 200; ++step)
        {
            if (rng() % 2 == 0 && pending.size() < 12)
            {
                stamp += 1 + static_cast<std::int64_t>(rng() % 50); // distinct, increasing
                const InFlightFrame f = frame(next++, stamp);
                c.submitted(f);
                pending.push_back(f);
            }
            else if (!pending.empty())
            {
                // Return a random in-flight frame; those before it are skipped by the encoder.
                const std::size_t k = rng() % pending.size();
                const auto r = c.returned(pending[k].submitted_us);
                REQUIRE(r);
                REQUIRE(r->identified);
                CHECK(r->frame.submitted_us == pending[k].submitted_us);
                CHECK(r->frame.sequence == pending[k].sequence);
                CHECK(r->skipped == k);
                pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(k) + 1);
            }
        }
        CHECK(c.in_flight() == pending.size());
    }
}
