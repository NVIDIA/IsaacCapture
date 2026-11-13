// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/timesync/local_clock.hpp"

#include <ctime>

namespace isaaccapture::timesync
{

namespace
{

int64_t read_clock_ns(clockid_t id)
{
    timespec ts{};
    int64_t ns = 0;
    if (clock_gettime(id, &ts) != 0 || __builtin_mul_overflow(static_cast<int64_t>(ts.tv_sec), 1'000'000'000LL, &ns) ||
        __builtin_add_overflow(ns, static_cast<int64_t>(ts.tv_nsec), &ns))
    {
        return kClockReadFailed;
    }
    return ns;
}

// Magnitude of a difference that cannot overflow, including INT64_MIN.
bool abs_diff(int64_t a, int64_t b, uint64_t& out)
{
    int64_t d = 0;
    if (__builtin_sub_overflow(a, b, &d))
    {
        return false;
    }
    out = d < 0 ? ~static_cast<uint64_t>(d) + 1 : static_cast<uint64_t>(d);
    return true;
}

} // namespace

ClockReader system_clock_reader()
{
    return { [] { return read_clock_ns(CLOCK_MONOTONIC); }, [] { return read_clock_ns(CLOCK_REALTIME); } };
}

LocalClockSample sample_local_clock(const ClockReader& reader)
{
    const int64_t before = reader.monotonic_ns();
    const int64_t wall = reader.realtime_ns();
    const int64_t after = reader.monotonic_ns();
    LocalClockSample s;
    s.read_uncertainty_ns = INT64_MAX;
    int64_t width = 0;
    if (before == kClockReadFailed || wall == kClockReadFailed || after == kClockReadFailed ||
        __builtin_sub_overflow(after, before, &width) || width < 0 || width == INT64_MAX)
    {
        return s;
    }
    s.monotonic_ns = before + width / 2;
    s.realtime_ns = wall;
    s.read_uncertainty_ns = (width + 1) / 2; // rounded up, so the pair is never claimed tighter than it is
    return s;
}

LocalClockTracker::LocalClockTracker(LocalClockConfig config) : config_(config)
{
}

LocalClockObservation LocalClockTracker::observe(const LocalClockSample& sample)
{
    LocalClockObservation obs;
    obs.sample = sample;
    obs.generation = generation_;
    if (sample.read_uncertainty_ns < 0 || sample.read_uncertainty_ns > config_.max_read_uncertainty_ns)
    {
        return obs; // unusable; the previous sample stays the reference
    }
    obs.usable = true;
    obs.step_tolerance_ns = config_.step_margin_ns;

    int64_t offset = 0;
    if (__builtin_sub_overflow(sample.realtime_ns, sample.monotonic_ns, &offset))
    {
        obs.usable = false;
        obs.step_tolerance_ns = 0;
        return obs;
    }

    if (have_previous_)
    {
        // CLOCK_MONOTONIC and CLOCK_REALTIME follow the same frequency adjustment, so their difference changes only
        // on a step (or a suspend, which pauses only the monotonic clock).
        uint64_t moved = 0;
        int64_t previous_offset = 0;
        if (__builtin_sub_overflow(previous_.realtime_ns, previous_.monotonic_ns, &previous_offset) ||
            !abs_diff(offset, previous_offset, moved))
        {
            obs.usable = false;
            obs.step_tolerance_ns = 0;
            return obs;
        }
        const __int128 allowed =
            static_cast<__int128>(sample.read_uncertainty_ns) + previous_.read_uncertainty_ns + config_.step_margin_ns;
        if (sample.monotonic_ns < previous_.monotonic_ns || static_cast<__int128>(moved) > allowed)
        {
            ++generation_;
            obs.stepped = true;
            obs.generation = generation_;
        }
    }
    previous_ = sample;
    have_previous_ = true;
    return obs;
}

} // namespace isaaccapture::timesync
