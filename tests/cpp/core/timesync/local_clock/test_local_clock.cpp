// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <timesync/local_clock.hpp>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace isaaccapture::timesync;

static int g_failures = 0;

#define CHECK(cond)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                              \
            ++g_failures;                                                                                              \
        }                                                                                                              \
    } while (0)

namespace
{

constexpr int64_t kSecond = 1'000'000'000;
constexpr int64_t kBase = 1'700'000'000'000'000'000;

LocalClockSample make_sample(int64_t mono, int64_t wall_minus_mono, int64_t uncertainty = 1000)
{
    LocalClockSample s;
    s.monotonic_ns = mono;
    s.realtime_ns = mono + wall_minus_mono;
    s.read_uncertainty_ns = uncertainty;
    return s;
}

void test_sample_local_clock()
{
    std::vector<int64_t> mono_script = { 1000, 1400 };
    size_t mono_calls = 0;
    std::string order;
    ClockReader reader;
    reader.monotonic_ns = [&]
    {
        order += 'm';
        return mono_script.at(mono_calls++);
    };
    reader.realtime_ns = [&]
    {
        order += 'r';
        return int64_t{ 5000 };
    };

    const LocalClockSample s = sample_local_clock(reader);
    CHECK(order == "mrm");
    CHECK(s.monotonic_ns == 1200);
    CHECK(s.realtime_ns == 5000);
    CHECK(s.read_uncertainty_ns == 200);
    CHECK(wall_minus_monotonic_ns(s) == 3800);
}

void test_system_reader_is_populated()
{
    const ClockReader r = system_clock_reader();
    CHECK(static_cast<bool>(r.monotonic_ns));
    CHECK(static_cast<bool>(r.realtime_ns));
    if (r.monotonic_ns && r.realtime_ns)
    {
        const LocalClockSample s = sample_local_clock(r);
        CHECK(s.read_uncertainty_ns >= 0);
        CHECK(s.realtime_ns > 0);
    }
}

void test_first_sample()
{
    LocalClockTracker tracker;
    const auto o = tracker.observe(make_sample(10 * kSecond, kBase));
    CHECK(o.usable);
    CHECK(o.generation == 1);
    CHECK(!o.stepped);
}

void test_steady_drift_does_not_step()
{
    LocalClockTracker tracker;
    int64_t offset = kBase;
    tracker.observe(make_sample(10 * kSecond, offset));
    for (int i = 1; i <= 20; ++i)
    {
        // 50 us of offset change per sample, inside the default step margin.
        offset += 50'000;
        const auto o = tracker.observe(make_sample((10 + i) * kSecond, offset));
        CHECK(o.usable);
        CHECK(!o.stepped);
        CHECK(o.generation == 1);
    }
}

void test_forward_and_backward_step()
{
    LocalClockTracker tracker;
    int64_t offset = kBase;
    tracker.observe(make_sample(10 * kSecond, offset));

    offset += 2 * kSecond; // forward step
    auto fwd = tracker.observe(make_sample(11 * kSecond, offset));
    CHECK(fwd.usable);
    CHECK(fwd.stepped);
    CHECK(fwd.generation == 2);

    auto steady = tracker.observe(make_sample(12 * kSecond, offset));
    CHECK(!steady.stepped);
    CHECK(steady.generation == 2);

    offset -= 5 * kSecond; // backward step
    auto back = tracker.observe(make_sample(13 * kSecond, offset));
    CHECK(back.usable);
    CHECK(back.stepped);
    CHECK(back.generation == 3);

    auto after = tracker.observe(make_sample(14 * kSecond, offset));
    CHECK(!after.stepped);
    CHECK(after.generation == 3);
}

void test_wide_bracket_is_unusable_and_harmless()
{
    const LocalClockConfig cfg; // max_read_uncertainty_ns = 200'000
    LocalClockTracker tracker(cfg);
    tracker.observe(make_sample(10 * kSecond, kBase));

    // Unusable sample whose offset would look like a step: no generation change, no step flag.
    auto wide = tracker.observe(make_sample(11 * kSecond, kBase + 3 * kSecond, cfg.max_read_uncertainty_ns + 1));
    CHECK(!wide.usable);
    CHECK(!wide.stepped);
    CHECK(wide.generation == 1);

    // The next narrow sample is compared with the last usable one, not with the wide one.
    auto next = tracker.observe(make_sample(12 * kSecond, kBase + 1000));
    CHECK(next.usable);
    CHECK(!next.stepped);
    CHECK(next.generation == 1);

    // A bracket exactly at the limit is still usable.
    auto edge = tracker.observe(make_sample(13 * kSecond, kBase + 1000, cfg.max_read_uncertainty_ns));
    CHECK(edge.usable);
    CHECK(edge.generation == 1);
}

void test_wide_bracket_does_not_hide_a_real_step()
{
    LocalClockTracker tracker;
    tracker.observe(make_sample(10 * kSecond, kBase));
    tracker.observe(make_sample(11 * kSecond, kBase, 5'000'000)); // unusable
    auto o = tracker.observe(make_sample(12 * kSecond, kBase + 4 * kSecond));
    CHECK(o.usable);
    CHECK(o.stepped);
    CHECK(o.generation == 2);
}

void test_small_jitter_within_uncertainty_is_not_a_step()
{
    LocalClockTracker tracker;
    tracker.observe(make_sample(10 * kSecond, kBase, 50'000));
    // Offset moves by 80 us: inside both bracket widths plus the margin.
    auto o = tracker.observe(make_sample(11 * kSecond, kBase + 80'000, 50'000));
    CHECK(!o.stepped);
    CHECK(o.generation == 1);
}

// Reader that replays scripted monotonic values around one realtime value.
ClockReader scripted_reader(std::vector<int64_t> mono, int64_t wall)
{
    auto calls = std::make_shared<size_t>(0);
    ClockReader r;
    r.monotonic_ns = [mono, calls] { return mono.at((*calls)++); };
    r.realtime_ns = [wall] { return wall; };
    return r;
}

void test_step_threshold_is_exact()
{
    const LocalClockConfig cfg;
    // Offset change `delta` between samples `gap` apart on the monotonic clock, with the given bracket widths.
    auto stepped = [&](int64_t delta, int64_t gap, int64_t width)
    {
        LocalClockTracker tracker(cfg);
        tracker.observe(make_sample(10 * kSecond, kBase, width));
        return tracker.observe(make_sample(10 * kSecond + gap, kBase + delta, width)).stepped;
    };
    for (int64_t gap : { int64_t{ 1'000'000 }, 1000 * kSecond })
    {
        // Zero widths: the threshold is the margin.
        CHECK(!stepped(cfg.step_margin_ns, gap, 0));
        CHECK(stepped(cfg.step_margin_ns + 1, gap, 0));
        CHECK(stepped(-cfg.step_margin_ns - 1, gap, 0));
        // Non-zero widths: the margin plus both of them.
        const int64_t threshold = cfg.step_margin_ns + 2 * 30'000;
        CHECK(!stepped(threshold, gap, 30'000));
        CHECK(stepped(threshold + 1, gap, 30'000));
    }
}

void test_long_idle_without_offset_change_is_not_a_step()
{
    LocalClockTracker tracker;
    tracker.observe(make_sample(10 * kSecond, kBase));
    const auto o = tracker.observe(make_sample(10 * kSecond + 100'000 * kSecond, kBase));
    CHECK(o.usable);
    CHECK(!o.stepped);
    CHECK(o.generation == 1);
}

void test_step_tolerance_reported()
{
    LocalClockConfig cfg;
    cfg.step_margin_ns = 123'456;
    LocalClockTracker tracker(cfg);
    CHECK(tracker.observe(make_sample(10 * kSecond, kBase)).step_tolerance_ns == cfg.step_margin_ns);
    CHECK(tracker.observe(make_sample(11 * kSecond, kBase + 2 * kSecond)).step_tolerance_ns == cfg.step_margin_ns);
    const auto wide = tracker.observe(make_sample(12 * kSecond, kBase, cfg.max_read_uncertainty_ns + 1));
    CHECK(!wide.usable);
    CHECK(wide.step_tolerance_ns == 0);
}

void test_failed_reads_are_unusable_and_harmless()
{
    std::vector<ClockReader> readers;
    readers.push_back(scripted_reader({ kClockReadFailed, 1400 }, 5000));
    readers.push_back(scripted_reader({ 1000, kClockReadFailed }, 5000));
    readers.push_back(scripted_reader({ 1000, 1400 }, kClockReadFailed));
    readers.push_back(scripted_reader({ 1000, 999 }, 5000)); // after = before - 1
    for (const ClockReader& reader : readers)
    {
        LocalClockTracker tracker;
        tracker.observe(make_sample(10 * kSecond, kBase));
        const LocalClockSample s = sample_local_clock(reader);
        CHECK(s.read_uncertainty_ns == INT64_MAX);
        const auto bad = tracker.observe(s);
        CHECK(!bad.usable);
        CHECK(!bad.stepped);
        CHECK(bad.generation == 1);
        // The reference is still the first sample.
        const auto next = tracker.observe(make_sample(11 * kSecond, kBase + 80'000));
        CHECK(next.usable);
        CHECK(!next.stepped);
        CHECK(next.generation == 1);
    }
}

void test_offset_overflow_is_unusable()
{
    {
        LocalClockTracker tracker;
        const LocalClockSample s = sample_local_clock(scripted_reader({ -5, -5 }, INT64_MAX));
        CHECK(s.read_uncertainty_ns == 0);
        const auto o = tracker.observe(s);
        CHECK(!o.usable);
        CHECK(!o.stepped);
        CHECK(o.generation == 1);
    }
    {
        // Overflow on a later sample leaves the earlier reference in place.
        LocalClockTracker tracker;
        tracker.observe(make_sample(10 * kSecond, kBase));
        const auto bad = tracker.observe(sample_local_clock(scripted_reader({ -5, -5 }, INT64_MAX)));
        CHECK(!bad.usable);
        CHECK(!bad.stepped);
        const auto next = tracker.observe(make_sample(11 * kSecond, kBase));
        CHECK(next.usable);
        CHECK(!next.stepped);
    }
}

void test_extreme_monotonic_values()
{
    auto raw = [](int64_t mono, int64_t wall)
    {
        LocalClockSample s;
        s.monotonic_ns = mono;
        s.realtime_ns = wall;
        s.read_uncertainty_ns = 0;
        return s;
    };
    {
        // Monotonic time far behind the previous sample with an unchanged offset: a backward step.
        LocalClockTracker tracker;
        tracker.observe(make_sample(10 * kSecond, kBase));
        const auto o = tracker.observe(raw(INT64_MIN, INT64_MIN + kBase));
        CHECK(o.usable);
        CHECK(o.stepped);
        CHECK(o.generation == 2);
    }
    {
        // Representable but enormous offset change.
        LocalClockTracker tracker;
        tracker.observe(make_sample(10 * kSecond, kBase));
        const auto o = tracker.observe(raw(INT64_MAX, INT64_MAX));
        CHECK(o.usable);
        CHECK(o.stepped);
        CHECK(o.generation == 2);
    }
    {
        // Offset difference overflows.
        LocalClockTracker tracker;
        tracker.observe(raw(0, INT64_MAX / 2 + 10));
        const auto o = tracker.observe(raw(1, -(INT64_MAX / 2) - 10));
        CHECK(!o.usable || o.stepped);
    }
}

} // namespace

int main()
{
    test_sample_local_clock();
    test_system_reader_is_populated();
    test_first_sample();
    test_steady_drift_does_not_step();
    test_forward_and_backward_step();
    test_wide_bracket_is_unusable_and_harmless();
    test_wide_bracket_does_not_hide_a_real_step();
    test_small_jitter_within_uncertainty_is_not_a_step();
    test_step_threshold_is_exact();
    test_long_idle_without_offset_change_is_not_a_step();
    test_step_tolerance_reported();
    test_failed_reads_are_unusable_and_harmless();
    test_offset_overflow_is_unusable();
    test_extreme_monotonic_values();
    if (g_failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("timesync_local_clock: all checks passed\n");
    return 0;
}
