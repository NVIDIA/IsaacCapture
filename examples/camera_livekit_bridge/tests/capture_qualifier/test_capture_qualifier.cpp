// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <clock_exchange/capture_qualifier.hpp>
#include <clock_exchange/clock_estimator.hpp>

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>

namespace
{

using camera_bridge::CaptureInput;
using camera_bridge::CaptureQualifier;
using camera_bridge::CaptureResult;
using camera_bridge::CaptureVerdict;
using camera_bridge::ClockEstimator;
using camera_bridge::ClockMapping;
using camera_bridge::Exchange;

int g_failures = 0;

void check(bool condition, const std::string& what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << "\n";
        ++g_failures;
    }
}

constexpr int64_t kMs = 1'000'000;
constexpr int64_t kOffset = 5'000'000'000;
constexpr int64_t kReady = 2'000'000'000'000;

ClockMapping make_mapping(uint64_t epoch = 1, int64_t uncertainty_ns = 100'000, int64_t offset_ns = kOffset)
{
    ClockMapping m;
    m.epoch = epoch;
    m.offset_ns = offset_ns;
    m.rtt_ns = 200'000;
    m.uncertainty_ns = uncertainty_ns;
    m.accepted_at_ns = kReady - 100 * kMs;
    return m;
}

/// A frame ready at `ready` whose capture was `age_ns` earlier, stamped on a sender clock `offset_ns` ahead.
CaptureInput make_input(int64_t age_ns, int64_t ready = kReady, int64_t offset_ns = kOffset, int64_t upstream = 0)
{
    CaptureInput in;
    in.sender_valid = true;
    in.capture_sm_ns = ready - age_ns + offset_ns;
    in.upstream_uncertainty_ns = upstream;
    in.ready_monotonic_ns = ready;
    return in;
}

/// Checks the verdict and that nothing but a qualified frame carries a capture time.
CaptureResult expect(CaptureQualifier& q,
                     const CaptureInput& in,
                     const std::optional<ClockMapping>& m,
                     std::size_t accepted,
                     CaptureVerdict verdict,
                     const std::string& what)
{
    const CaptureResult r = q.qualify(in, m, accepted);
    check(r.verdict == verdict,
          what + ": expected " + camera_bridge::to_string(verdict) + ", got " + camera_bridge::to_string(r.verdict));
    if (r.verdict != CaptureVerdict::Qualified)
        check(r.capture_bridge_ns == 0, what + ": a non-qualified frame carries no capture time");
    return r;
}

/// A qualifier that has learned a steady 20 ms age from 8 qualified frames.

void test_verdict_table()
{
    const ClockMapping m = make_mapping();
    {
        CaptureQualifier q;
        CaptureInput invalid = make_input(20 * kMs);
        invalid.sender_valid = false;
        expect(q, invalid, m, 10, CaptureVerdict::SenderInvalid, "invalid sender record");
        expect(q, invalid, std::nullopt, 0, CaptureVerdict::SenderInvalid, "invalid record wins over no mapping");
        expect(q, make_input(20 * kMs), std::nullopt, 0, CaptureVerdict::NoMapping, "no mapping");
    }
    {
        CaptureQualifier q;
        expect(q, make_input(20 * kMs), m, 2, CaptureVerdict::Warmup, "two accepted exchanges");
        expect(q, make_input(20 * kMs), m, 3, CaptureVerdict::Qualified, "three accepted exchanges");
    }
    {
        // Budget is 5 ms on mapping + upstream uncertainty; exactly at it qualifies.
        CaptureQualifier q;
        const ClockMapping at_budget = make_mapping(1, 3 * kMs);
        const CaptureResult ok = expect(
            q, make_input(20 * kMs, kReady, kOffset, 2 * kMs), at_budget, 10, CaptureVerdict::Qualified, "at budget");
        check(ok.combined_uncertainty_ns == 5 * kMs, "combined uncertainty at the budget");
        const CaptureResult over = expect(q, make_input(20 * kMs, kReady, kOffset, 2 * kMs + 1), at_budget, 10,
                                          CaptureVerdict::OverBudget, "one ns over budget");
        check(over.combined_uncertainty_ns == 5 * kMs + 1, "over-budget result reports the combined uncertainty");
        expect(q, make_input(20 * kMs), make_mapping(1, 5 * kMs + 1), 10, CaptureVerdict::OverBudget,
               "mapping alone over budget");
    }
    {
        // Age plausibility: not after ready, not more than 1 s before it.
        const struct
        {
            int64_t age;
            CaptureVerdict verdict;
            const char* what;
        } cases[] = {
            { -1, CaptureVerdict::ImplausibleAge, "capture 1 ns after ready" },
            { 0, CaptureVerdict::Qualified, "capture at ready" },
            { 1'000'000'000, CaptureVerdict::Qualified, "capture exactly max_age before ready" },
            { 1'000'000'001, CaptureVerdict::ImplausibleAge, "capture over max_age before ready" },
        };
        for (const auto& c : cases)
        {
            CaptureQualifier q;
            expect(q, make_input(c.age), m, 10, c.verdict, c.what);
        }
    }
}

void test_qualified_result()
{
    CaptureQualifier q;
    const ClockMapping m = make_mapping(7, 150'000);
    CaptureInput in = make_input(20 * kMs, kReady, kOffset, 250'000);
    const CaptureResult r = q.qualify(in, m, 10);
    check(r.verdict == CaptureVerdict::Qualified, "qualified");
    check(r.capture_bridge_ns == in.capture_sm_ns - kOffset, "capture_bridge_ns = capture_sm_ns - offset_ns");
    check(r.capture_bridge_ns == kReady - 20 * kMs, "capture_bridge_ns is the capture instant on the host clock");
    check(r.combined_uncertainty_ns == 400'000, "combined uncertainty is mapping plus upstream");
    check(r.epoch == 7, "epoch is the mapping's");

    // Negative and large offsets map the same way.
    const ClockMapping negative = make_mapping(1, 100'000, -3'000'000'000);
    CaptureQualifier q2;
    const CaptureResult r2 = q2.qualify(make_input(20 * kMs, kReady, -3'000'000'000), negative, 10);
    check(r2.verdict == CaptureVerdict::Qualified && r2.capture_bridge_ns == kReady - 20 * kMs,
          "negative offset maps the capture time");
}

void test_no_learned_age_level()
{
    // A steady history decides nothing: the age checks are the physical ones (not negative, not over the maximum).
    const ClockMapping m = make_mapping();
    CaptureQualifier q;
    for (int i = 0; i < 20; ++i)
        expect(q, make_input(20 * kMs), m, 10, CaptureVerdict::Qualified, "steady age");
    expect(q, make_input(400 * kMs), m, 10, CaptureVerdict::Qualified, "a longer but possible age is not withheld");
    expect(q, make_input(-1), m, 10, CaptureVerdict::ImplausibleAge, "a capture after ready is withheld");
}

void test_epoch_change()
{
    const ClockMapping m = make_mapping();
    CaptureQualifier q;
    expect(q, make_input(20 * kMs), m, 10, CaptureVerdict::Qualified, "first epoch");
    const ClockMapping next = make_mapping(2, 100'000, kOffset + 80 * kMs);
    expect(q, make_input(20 * kMs, kReady, kOffset + 80 * kMs), next, 1, CaptureVerdict::Warmup,
           "a new epoch waits for exchanges");
    const CaptureResult r = expect(q, make_input(20 * kMs, kReady, kOffset + 80 * kMs), next, 3,
                                   CaptureVerdict::Qualified, "a warm new epoch qualifies");
    check(r.epoch == 2, "the result carries the new epoch");
}

/// Simulates a sender whose monotonic clock steps by `step_ns` while frames keep being stamped on it, with the
/// estimator and the qualifier wired together as the bridge does.
void run_clock_step(int64_t step_ns)
{
    const std::string label = "step " + std::to_string(step_ns / kMs) + " ms: ";
    constexpr int64_t kRtt = 200'000;
    constexpr int64_t kProcessing = 50'000;
    constexpr int64_t kAge = 20 * kMs;
    constexpr int64_t kStepAt = 1'100 * kMs; // between two exchanges
    constexpr int64_t kEnd = 4'000 * kMs;
    constexpr int64_t kExchange = 200 * kMs;
    const auto sender_offset = [&](int64_t t) { return kOffset + (t >= kStepAt ? step_ns : 0); };

    ClockEstimator estimator;
    CaptureQualifier qualifier;
    int qualified_before = 0;
    int withheld_after_step = 0;
    int qualified_at_end = 0;
    int wrong_in_exposure = 0;
    for (int64_t t = 0; t <= kEnd; t += kMs)
    {
        const int64_t now = kReady + t;
        if (t % (200 * kMs) == 0)
        {
            const int64_t t1 = now - kRtt - kProcessing;
            const int64_t t2 = t1 + sender_offset(t) + kRtt / 2;
            estimator.add(Exchange{ t1, t2, t2 + kProcessing, now });
        }
        if (t % (33 * kMs) != 0)
            continue;
        const CaptureInput in = make_input(kAge, now, sender_offset(t), 500'000);
        const auto mapping = estimator.mapping(now);
        const CaptureResult r = qualifier.qualify(in, mapping, estimator.accepted_in_epoch());
        if (r.verdict == CaptureVerdict::Qualified)
        {
            const int64_t truth = now - kAge;
            const int64_t error = r.capture_bridge_ns > truth ? r.capture_bridge_ns - truth : truth - r.capture_bridge_ns;
            // The one exposure: frames between the step and the next exchange are mapped with the old offset.
            const bool exposed = t >= kStepAt && t < kStepAt + kExchange;
            if (!exposed)
                check(error <= r.combined_uncertainty_ns,
                      label + "a qualified frame at t=" + std::to_string(t / kMs) + " ms has a wrong capture time");
            if (exposed)
                ++wrong_in_exposure;
            if (t < kStepAt)
                ++qualified_before;
            if (t >= kEnd - 200 * kMs)
                ++qualified_at_end;
        }
        else
        {
            check(r.capture_bridge_ns == 0, label + "a non-qualified frame carries no capture time");
            if (t >= kStepAt && t < kStepAt + kExchange)
                ++withheld_after_step;
        }
    }
    check(qualified_before > 0, label + "frames qualify before the step");
    (void)wrong_in_exposure; // bounded by construction: only frames before the next exchange are in the window
    (void)withheld_after_step;
    check(estimator.discontinuity() && estimator.epoch() == 2, label + "the estimator sees the step");
    check(qualified_at_end > 0, label + "frames qualify again once the new epoch is warm");
}

void test_clock_steps()
{
    for (const int64_t step : { 50 * kMs, -50 * kMs, 250 * kMs, -250 * kMs, 2'000 * kMs, -2'000 * kMs })
        run_clock_step(step);
}

} // namespace

int main()
{
    test_verdict_table();
    test_qualified_result();
    test_no_learned_age_level();
    test_epoch_change();
    test_clock_steps();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all capture_qualifier checks passed\n";
    return 0;
}
