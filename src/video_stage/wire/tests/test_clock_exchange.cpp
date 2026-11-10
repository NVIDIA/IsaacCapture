// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <video_stage/clock_exchange.hpp>

#include <cstdlib>

using video_stage::ClockEstimator;
using video_stage::Exchange;

namespace
{
constexpr int64_t MS = 1'000'000;
constexpr int64_t S = 1'000'000'000;

// An exchange between a local clock and a remote clock running `offset` ahead of it.
Exchange make_exchange(int64_t t1, int64_t offset = 0, int64_t forward = MS, int64_t back = MS, int64_t processing = 0)
{
    const int64_t arrival = t1 + forward;
    const int64_t departure = arrival + processing;
    return Exchange{ t1, arrival + offset, departure + offset, departure + back };
}
} // namespace

TEST_CASE("symmetric delays recover offset, rtt excludes processing", "[Clock]")
{
    const Exchange ex = make_exchange(10 * S, 123'456'000, 3 * MS, 3 * MS, 40 * MS);
    CHECK(ex.offset_ns() == 123'456'000);
    CHECK(ex.rtt_ns() == 6 * MS);

    ClockEstimator est;
    CHECK(est.add(ex).empty());
    const auto m = est.mapping(ex.t4);
    REQUIRE(m);
    CHECK(m->offset_ns == 123'456'000);
    CHECK(m->rtt_ns == 6 * MS);
    CHECK(m->uncertainty_ns == 3 * MS);
    CHECK(m->to_local_ns(S + 123'456'000) == S);
}

TEST_CASE("asymmetric delay error is half the difference, within uncertainty", "[Clock]")
{
    const int64_t pairs[][2] = { { 1 * MS, 9 * MS }, { 9 * MS, 1 * MS }, { 2 * MS, 2 * MS } };
    for (const auto& p : pairs)
    {
        const Exchange ex = make_exchange(S, 5 * MS, p[0], p[1], 7 * MS);
        ClockEstimator est;
        est.add(ex);
        const auto m = est.mapping(ex.t4);
        REQUIRE(m);
        const int64_t error = m->offset_ns - 5 * MS;
        CHECK(error == (p[0] - p[1]) / 2);
        CHECK(std::abs(error) <= m->uncertainty_ns);
    }
}

TEST_CASE("negative offset is recovered", "[Clock]")
{
    const Exchange ex = make_exchange(20 * S, -250 * MS, 2 * MS, 2 * MS, MS);
    ClockEstimator est;
    est.add(ex);
    CHECK(est.mapping(ex.t4)->offset_ns == -250 * MS);
}

TEST_CASE("malformed and slow exchanges are rejected without changing the mapping", "[Clock]")
{
    const Exchange bad[] = {
        make_exchange(S, 0, 200 * MS, 100 * MS), // rtt 300 ms
        Exchange{ 100, 0, 0, 50 }, // t4 before t1
        Exchange{ 0, 100, 50, 1000 }, // t3 before t2
        Exchange{ 0, 0, 900, 800 }, // negative rtt
    };
    for (const Exchange& b : bad)
    {
        ClockEstimator est;
        est.add(make_exchange(S, MS));
        const auto before = est.mapping(2 * S);
        REQUIRE(before);
        CHECK_FALSE(est.add(b).empty());
        const auto after = est.mapping(2 * S);
        REQUIRE(after);
        CHECK(after->offset_ns == before->offset_ns);
        CHECK(after->rtt_ns == before->rtt_ns);
        CHECK(after->accepted_at_ns == before->accepted_at_ns);
        CHECK(after->uncertainty_ns == before->uncertainty_ns);
        CHECK_FALSE(est.discontinuity());
    }
}

TEST_CASE("best of the window is the lowest rtt", "[Clock]")
{
    ClockEstimator est;
    const int64_t delays[] = { 5 * MS, 3 * MS, 8 * MS, 2 * MS, 6 * MS };
    int64_t t = S;
    for (int64_t d : delays)
    {
        est.add(make_exchange(t, MS, d, d));
        t += S;
    }
    CHECK(est.mapping(6 * S)->rtt_ns == 4 * MS);
}

TEST_CASE("high rtt outlier does not replace the better mapping; uncertainty ages at read", "[Clock]")
{
    ClockEstimator est;
    est.add(make_exchange(1 * S, 10 * MS, MS, MS));
    const auto good = est.mapping(2 * S);
    REQUIRE(good);
    est.add(make_exchange(2 * S, 10 * MS, 20 * MS, 20 * MS));
    const auto m = est.mapping(3 * S);
    REQUIRE(m);
    CHECK(m->rtt_ns == good->rtt_ns);
    CHECK(m->accepted_at_ns == good->accepted_at_ns);
    CHECK(m->uncertainty_ns == good->rtt_ns / 2 + static_cast<int64_t>(200e-6 * (3 * S - good->accepted_at_ns)));
}

TEST_CASE("uncertainty ages between exchanges without a new one", "[Clock]")
{
    ClockEstimator est;
    const Exchange ex = make_exchange(10 * S, 5 * MS, 2 * MS, 2 * MS);
    est.add(ex);
    const auto at_add = est.mapping(ex.t4);
    REQUIRE(at_add);
    CHECK(at_add->uncertainty_ns == ex.rtt_ns() / 2);
    const auto later = est.mapping(ex.t4 + 3 * S);
    REQUIRE(later);
    CHECK(later->uncertainty_ns == ex.rtt_ns() / 2 + static_cast<int64_t>(200e-6 * 3 * S));
    CHECK(later->offset_ns == at_add->offset_ns);
    CHECK(later->accepted_at_ns == at_add->accepted_at_ns);
}

TEST_CASE("steady link keeps choosing the newest exchange", "[Clock]")
{
    ClockEstimator est;
    for (int s = 1; s <= 30; ++s)
    {
        const Exchange ex = make_exchange(s * S, 7 * MS, 2 * MS, 2 * MS);
        CHECK(est.add(ex).empty());
        const auto m = est.mapping(ex.t4);
        REQUIRE(m);
        CHECK(m->accepted_at_ns == ex.t4);
        CHECK(m->uncertainty_ns == 2 * MS);
    }
    CHECK_FALSE(est.discontinuity());
}

TEST_CASE("mapping is none before any exchange and after the stale window", "[Clock]")
{
    ClockEstimator est;
    CHECK_FALSE(est.mapping(0));
    const Exchange ex = make_exchange(S, MS);
    est.add(ex);
    CHECK(est.mapping(ex.t4));
    CHECK(est.mapping(ex.t4 + 5 * S));
    CHECK_FALSE(est.mapping(ex.t4 + 5 * S + 1));

    // A new exchange brings it back.
    est.add(make_exchange(70 * S, MS));
    CHECK(est.mapping(70 * S + 2 * MS));
}

TEST_CASE("rejected exchanges do not refresh staleness", "[Clock]")
{
    ClockEstimator est;
    const Exchange ex = make_exchange(S, MS);
    est.add(ex);
    const int64_t late = ex.t4 + 5 * S + 1;
    REQUIRE_FALSE(est.mapping(late));
    CHECK_FALSE(est.add(make_exchange(3 * S, MS, 200 * MS, 100 * MS)).empty());
    CHECK_FALSE(est.add(Exchange{ 6 * S, 0, 0, 5 * S }).empty());
    CHECK_FALSE(est.mapping(late));
}

TEST_CASE("a clock step is a discontinuity and restarts from the new exchange", "[Clock]")
{
    ClockEstimator est;
    for (int s = 1; s <= 5; ++s)
        est.add(make_exchange(s * S, 0, 2 * MS, 2 * MS));
    CHECK_FALSE(est.discontinuity());

    // Same rtt as before: the step alone must flag it.
    const Exchange step = make_exchange(6 * S, 50 * MS, 2 * MS, 2 * MS);
    CHECK(est.add(step).empty());
    CHECK(est.discontinuity());
    const auto m = est.mapping(step.t4);
    REQUIRE(m);
    CHECK(m->offset_ns == 50 * MS);
    CHECK(m->accepted_at_ns == step.t4);

    // Sticky until cleared; later good exchanges do not clear it.
    est.add(make_exchange(7 * S, 50 * MS, MS, MS));
    CHECK(est.discontinuity());
    CHECK(est.mapping(7 * S + 2 * MS)->offset_ns == 50 * MS);

    est.clear_discontinuity();
    CHECK_FALSE(est.discontinuity());
    est.add(make_exchange(8 * S, 50 * MS, MS, MS));
    CHECK_FALSE(est.discontinuity());
}

TEST_CASE("a lower-rtt step does not mix old exchanges into the new mapping", "[Clock]")
{
    ClockEstimator est;
    for (int s = 1; s <= 3; ++s)
        est.add(make_exchange(s * S, 0, 2 * MS, 2 * MS));
    est.add(make_exchange(4 * S, 50 * MS, MS, MS));
    CHECK(est.discontinuity());
    CHECK(est.mapping(4 * S + 2 * MS)->offset_ns == 50 * MS);
}

TEST_CASE("drift stays continuous and tracks within uncertainty plus drift", "[Clock]")
{
    ClockEstimator est;
    const double ppm = 50.0;
    for (int s = 1; s <= 30; ++s)
    {
        const int64_t t1 = s * S;
        auto remote = [&](int64_t local) { return local + static_cast<int64_t>(local * ppm * 1e-6); };
        const Exchange ex{ t1, remote(t1 + MS), remote(t1 + MS), t1 + 2 * MS };
        CHECK(est.add(ex).empty());
        const auto m = est.mapping(ex.t4);
        REQUIRE(m);
        const double true_offset = ppm * 1e-6 * ex.t4;
        const double drift = ppm * 1e-6 * (ex.t4 - m->accepted_at_ns);
        CHECK(std::abs(m->offset_ns - true_offset) <= m->uncertainty_ns + drift + 2);
    }
    CHECK_FALSE(est.discontinuity());
}
