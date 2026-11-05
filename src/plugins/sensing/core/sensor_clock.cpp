// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sensor_clock.hpp"

#include <ctime>

namespace plugins
{
namespace sensing
{

namespace
{

int64_t clock_ns(clockid_t id)
{
    timespec ts{};
    clock_gettime(id, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

#if defined(__aarch64__)
int64_t read_counter()
{
    uint64_t v;
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v)::"memory");
    return static_cast<int64_t>(v);
}
#endif

/// The narrowest of several back-to-back reads: a descheduled read has a wide bracket and is discarded.
template <typename ReadSource>
ClockPair narrowest_pair(ReadSource read_source)
{
    constexpr int kAttempts = 8;
    ClockPair best{};
    best.bracket_ns = INT64_MAX;
    for (int i = 0; i < kAttempts; ++i)
    {
        const int64_t before = clock_ns(CLOCK_MONOTONIC);
        const int64_t source = read_source();
        const int64_t after = clock_ns(CLOCK_MONOTONIC);
        if (after - before < best.bracket_ns)
            best = { source, before + (after - before) / 2, after - before };
    }
    return best;
}

} // namespace

ClockPair read_counter_pair()
{
#if defined(__aarch64__)
    return narrowest_pair(read_counter);
#else
    throw std::runtime_error("the ARM generic counter can only be read on aarch64");
#endif
}

int64_t read_counter_hz()
{
#if defined(__aarch64__)
    uint64_t hz;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(hz));
    return static_cast<int64_t>(hz);
#else
    throw std::runtime_error("the ARM generic counter can only be read on aarch64");
#endif
}

ClockPair read_raw_pair()
{
    return narrowest_pair([] { return clock_ns(CLOCK_MONOTONIC_RAW); });
}

} // namespace sensing
} // namespace plugins
