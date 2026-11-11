// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Golden numbers were produced by running egocap's Python ClockEstimator
// (examples/egocap/backend/egocap_web/clock_exchange.py) over the sequences built by make_exchange() below;
// the test does not run Python.

#include <clock_exchange/clock_estimator.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace
{

using camera_bridge::ClockEstimator;
using camera_bridge::ClockMapping;
using camera_bridge::Exchange;

int g_failures = 0;

void check(bool condition, const char* what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << "\n";
        ++g_failures;
    }
}

constexpr int64_t kT0 = 1'000'000'000'000;
constexpr int64_t kSpacing = 200'000'000;
constexpr int64_t kProcessing = 50'000;
constexpr int64_t kOffset = 5'000'000'000;

/// Exchange i of a sequence: t1 local, t2 = t1 + offset + rtt/2 (symmetric path), t3 = t2 + processing time,
/// t4 = t1 + rtt + processing time.
Exchange make_exchange(int i, int64_t rtt_ns, int64_t offset_ns)
{
    const int64_t t1 = kT0 + i * kSpacing;
    const int64_t t2 = t1 + offset_ns + rtt_ns / 2;
    return Exchange{ t1, t2, t2 + kProcessing, t1 + rtt_ns + kProcessing };
}

struct Golden
{
    uint64_t epoch;
    int64_t offset_ns;
    int64_t rtt_ns;
    int64_t uncertainty_ns;
    int64_t accepted_at_ns;
};

bool matches(const std::optional<ClockMapping>& m, const Golden& g)
{
    return m && m->epoch == g.epoch && m->offset_ns == g.offset_ns && m->rtt_ns == g.rtt_ns &&
           m->uncertainty_ns == g.uncertainty_ns && m->accepted_at_ns == g.accepted_at_ns;
}

// Sequence A: 12 exchanges, rtt 200 us, offset 5e9 ns, 200 ms apart; mapping(t4) after each.
const std::array<Golden, 12> kGoldenA = { {
    { 1, 5000000000, 200000, 100000, 1000000250000 },
    { 1, 5000000000, 200000, 100000, 1000200250000 },
    { 1, 5000000000, 200000, 100000, 1000400250000 },
    { 1, 5000000000, 200000, 100000, 1000600250000 },
    { 1, 5000000000, 200000, 100000, 1000800250000 },
    { 1, 5000000000, 200000, 100000, 1001000250000 },
    { 1, 5000000000, 200000, 100000, 1001200250000 },
    { 1, 5000000000, 200000, 100000, 1001400250000 },
    { 1, 5000000000, 200000, 100000, 1001600250000 },
    { 1, 5000000000, 200000, 100000, 1001800250000 },
    { 1, 5000000000, 200000, 100000, 1002000250000 },
    { 1, 5000000000, 200000, 100000, 1002200250000 },
} };

// Sequence B: as A, but exchange 2 has rtt 40 us, so best-of-window keeps it (the mapping keeps its
// accepted_at_ns) until the drift allowance over its age makes a fresh 200 us exchange cheaper.
const std::array<int64_t, 12> kRttsB = { 200000, 200000, 40000,  200000, 200000, 200000,
                                         200000, 200000, 200000, 200000, 200000, 200000 };
const std::array<Golden, 12> kGoldenB = { {
    { 1, 5000000000, 200000, 100000, 1000000250000 },
    { 1, 5000000000, 200000, 100000, 1000200250000 },
    { 1, 5000000000, 40000, 20000, 1000400090000 },
    { 1, 5000000000, 40000, 60032, 1000400090000 },
    { 1, 5000000000, 200000, 100000, 1000800250000 },
    { 1, 5000000000, 200000, 100000, 1001000250000 },
    { 1, 5000000000, 200000, 100000, 1001200250000 },
    { 1, 5000000000, 200000, 100000, 1001400250000 },
    { 1, 5000000000, 200000, 100000, 1001600250000 },
    { 1, 5000000000, 200000, 100000, 1001800250000 },
    { 1, 5000000000, 200000, 100000, 1002000250000 },
    { 1, 5000000000, 200000, 100000, 1002200250000 },
} };

// Sequence C: six exchanges at offset 5e9, then three at 5.3e9 (a +300 ms jump).
const std::array<Golden, 9> kGoldenC = { {
    { 1, 5000000000, 200000, 100000, 1000000250000 },
    { 1, 5000000000, 200000, 100000, 1000200250000 },
    { 1, 5000000000, 200000, 100000, 1000400250000 },
    { 1, 5000000000, 200000, 100000, 1000600250000 },
    { 1, 5000000000, 200000, 100000, 1000800250000 },
    { 1, 5000000000, 200000, 100000, 1001000250000 },
    { 2, 5300000000, 200000, 100000, 1001200250000 },
    { 2, 5300000000, 200000, 100000, 1001400250000 },
    { 2, 5300000000, 200000, 100000, 1001600250000 },
} };

void test_clean_sequence()
{
    ClockEstimator est;
    check(!est.mapping(kT0).has_value(), "no mapping before the first exchange");
    for (int i = 0; i < 12; ++i)
    {
        const Exchange x = make_exchange(i, 200'000, kOffset);
        check(!est.add(x).has_value(), "clean exchange accepted");
        check(matches(est.mapping(x.t4), kGoldenA[i]), "sequence A mapping equals the Python estimator's");
        check(est.accepted_in_epoch() == static_cast<std::size_t>(i + 1), "accepted_in_epoch counts exchanges");
        check(!est.discontinuity(), "clean sequence has no discontinuity");
        check(est.epoch() == 1, "clean sequence stays in epoch 1");
    }
}

void test_best_of_window()
{
    ClockEstimator est;
    for (int i = 0; i < 12; ++i)
    {
        const Exchange x = make_exchange(i, kRttsB[i], kOffset);
        check(!est.add(x).has_value(), "exchange B accepted");
        check(matches(est.mapping(x.t4), kGoldenB[i]), "sequence B mapping equals the Python estimator's");
    }
}

void test_rejections()
{
    ClockEstimator est;
    const auto slow = est.add(make_exchange(0, 300'000'000, kOffset));
    check(slow && *slow == "round trip too long", "slow exchange rejected as too long");
    check(!est.mapping(kT0 + 400'000'000).has_value() && est.accepted_in_epoch() == 0,
          "a rejected exchange leaves no mapping");

    const Exchange good = make_exchange(0, 200'000, kOffset);
    const auto t4_before_t1 = est.add(Exchange{ good.t1, good.t2, good.t3, good.t1 - 1 });
    check(t4_before_t1 && *t4_before_t1 == "malformed exchange", "t4 < t1 is malformed");
    const auto t3_before_t2 = est.add(Exchange{ good.t1, good.t2, good.t2 - 1, good.t4 });
    check(t3_before_t2 && *t3_before_t2 == "malformed exchange", "t3 < t2 is malformed");
    const auto negative_rtt = est.add(Exchange{ 0, 0, 500, 100 });
    check(negative_rtt && *negative_rtt == "malformed exchange", "negative round trip is malformed");
    check(est.accepted_in_epoch() == 0 && !est.mapping(good.t4).has_value(), "malformed exchanges leave no mapping");

    check(!est.add(make_exchange(0, 250'000'000, kOffset)).has_value(), "rtt exactly at the limit is accepted");
    const auto over = est.add(make_exchange(1, 250'000'001, kOffset));
    check(over && *over == "round trip too long", "rtt one ns over the limit is rejected");
}

void test_offset_jump()
{
    ClockEstimator est;
    for (int i = 0; i < 9; ++i)
    {
        const Exchange x = make_exchange(i, 200'000, i < 6 ? kOffset : kOffset + 300'000'000);
        check(!est.add(x).has_value(), "exchange C accepted");
        check(matches(est.mapping(x.t4), kGoldenC[i]), "sequence C mapping equals the Python estimator's");
        check(est.discontinuity() == (i >= 6), "discontinuity set from the jump on and left set");
        check(est.epoch() == (i >= 6 ? 2U : 1U), "epoch increments once at the jump");
        const std::size_t expected = i < 6 ? static_cast<std::size_t>(i + 1) : static_cast<std::size_t>(i - 5);
        check(est.accepted_in_epoch() == expected, "accepted_in_epoch restarts at the jump");
    }

    // A step inside the uncertainties (100 us) is a refinement, not a discontinuity (Python: epoch 1, offset
    // 5000100000).
    ClockEstimator small;
    for (int i = 0; i < 3; ++i)
        small.add(make_exchange(i, 200'000, kOffset));
    small.add(make_exchange(3, 200'000, kOffset + 100'000));
    check(!small.discontinuity() && small.epoch() == 1, "a step within the uncertainty is not a discontinuity");
    const auto m = small.mapping(make_exchange(3, 200'000, 0).t4);
    check(m && m->offset_ns == 5'000'100'000, "the refined offset is adopted");
}

void test_staleness_and_aging()
{
    ClockEstimator est;
    for (int i = 0; i < 12; ++i)
        est.add(make_exchange(i, 200'000, kOffset));
    const int64_t last = make_exchange(11, 200'000, 0).t4;

    // Golden uncertainties from the Python estimator at last + dt: 200 ppm of the age on top of rtt/2.
    struct Aged
    {
        int64_t dt;
        int64_t uncertainty;
    };
    for (const Aged a : { Aged{ 0, 100000 }, Aged{ 1'000'000'000, 300000 }, Aged{ 2'500'000'000, 600000 },
                          Aged{ 5'000'000'000, 1100000 } })
    {
        const auto m = est.mapping(last + a.dt);
        check(m && m->uncertainty_ns == a.uncertainty && m->accepted_at_ns == last && m->offset_ns == kOffset,
              "uncertainty grows with age by 200 ppm");
    }
    check(!est.mapping(last + 5'000'000'001).has_value(), "no mapping more than 5 s after the last exchange");
    const auto before = est.mapping(last - 1'000'000);
    check(before && before->uncertainty_ns == 100'000, "a query before the exchange does not shrink the uncertainty");

    // A rejected exchange does not refresh staleness.
    est.add(make_exchange(30, 300'000'000, kOffset));
    check(!est.mapping(last + 5'000'000'001).has_value(), "a rejected exchange does not keep the mapping alive");
}

void test_reset()
{
    ClockEstimator est;
    for (int i = 0; i < 5; ++i)
        est.add(make_exchange(i, 200'000, kOffset));
    est.add(make_exchange(5, 200'000, kOffset + 300'000'000));
    check(est.discontinuity() && est.epoch() == 2, "setup: discontinuity in epoch 2");

    est.reset();
    check(!est.mapping(make_exchange(5, 0, 0).t4).has_value(), "reset clears the mapping");
    check(est.epoch() == 3, "reset bumps the epoch");
    check(est.accepted_in_epoch() == 0, "reset clears accepted_in_epoch");
    // This C++ reset also clears the discontinuity flag; egocap's Python reset leaves it set.
    check(!est.discontinuity(), "reset clears the discontinuity flag (C++ behaviour)");

    const Exchange next = make_exchange(6, 200'000, kOffset);
    check(!est.add(next).has_value(), "exchange after reset accepted");
    const auto m = est.mapping(next.t4);
    check(m && m->epoch == 3 && m->offset_ns == kOffset && est.accepted_in_epoch() == 1,
          "the new epoch starts from the first exchange after reset");
}

void test_floor_division()
{
    using camera_bridge::floor_div;
    // Python: -1 // 2 == -1, -3 // 2 == -2, -5 // 2 == -3, -7 // 2 == -4, -1000000001 // 2 == -500000001,
    // -5 // 3 == -2, -1000000001 // 3 == -333333334, 7 // 2 == 3, 5 // -2 == -3, -4 // 2 == -2, 0 // 2 == 0.
    check(floor_div(-1, 2) == -1, "floor_div(-1, 2)");
    check(floor_div(-3, 2) == -2, "floor_div(-3, 2)");
    check(floor_div(-5, 2) == -3, "floor_div(-5, 2)");
    check(floor_div(-7, 2) == -4, "floor_div(-7, 2)");
    check(floor_div(-1'000'000'001, 2) == -500'000'001, "floor_div(-1000000001, 2)");
    check(floor_div(-5, 3) == -2, "floor_div(-5, 3)");
    check(floor_div(-1'000'000'001, 3) == -333'333'334, "floor_div(-1000000001, 3)");
    check(floor_div(7, 2) == 3, "floor_div(7, 2)");
    check(floor_div(5, -2) == -3, "floor_div(5, -2)");
    check(floor_div(-4, 2) == -2, "floor_div(-4, 2)");
    check(floor_div(0, 2) == 0, "floor_div(0, 2)");

    // Python: Exchange(0,-4,-3,2).offset_ns == -5, .rtt_ns == 1; Exchange(0,5,6,3).offset_ns == 4.
    check(Exchange{ 0, -4, -3, 2 }.offset_ns() == -5, "negative odd offset rounds down");
    check(Exchange{ 0, -4, -3, 2 }.rtt_ns() == 1, "rtt of the negative-offset exchange");
    check(Exchange{ 0, 5, 6, 3 }.offset_ns() == 4, "positive odd offset");
    check(Exchange{ 0, -4, -3, 0 }.offset_ns() == -4, "offset of a malformed exchange still floors");
    check(!Exchange{ 0, -4, -3, 0 }.well_formed(), "negative round trip is not well formed");
}

} // namespace

int main()
{
    test_clean_sequence();
    test_best_of_window();
    test_rejections();
    test_offset_jump();
    test_staleness_and_aging();
    test_reset();
    test_floor_division();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all clock_estimator checks passed\n";
    return 0;
}
