// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>

namespace isaaccapture::timesync
{

/// Where a host reads its clocks. Tests substitute fakes; the default reads CLOCK_MONOTONIC and CLOCK_REALTIME.
/// A failed read returns kClockReadFailed.
inline constexpr int64_t kClockReadFailed = INT64_MIN;

struct ClockReader
{
    std::function<int64_t()> monotonic_ns;
    std::function<int64_t()> realtime_ns;
};

ClockReader system_clock_reader();

/// One paired reading: the realtime value is taken between two monotonic reads, and the monotonic instant is the
/// bracket midpoint, so the pair is off by at most `read_uncertainty_ns`. A failed or inconsistent read gives
/// `read_uncertainty_ns == INT64_MAX`, which no tracker accepts.
struct LocalClockSample
{
    int64_t monotonic_ns = 0;
    int64_t realtime_ns = 0;
    int64_t read_uncertainty_ns = 0;
};

LocalClockSample sample_local_clock(const ClockReader& reader);

/// realtime minus monotonic at the sample, the offset that maps a realtime stamp onto this host's monotonic clock.
constexpr int64_t wall_minus_monotonic_ns(const LocalClockSample& s)
{
    return s.realtime_ns - s.monotonic_ns;
}

struct LocalClockConfig
{
    /// A sample whose bracket is wider than this is unusable.
    int64_t max_read_uncertainty_ns = 200'000;
    /// CLOCK_REALTIME and CLOCK_MONOTONIC share frequency adjustment on Linux, so their difference changes only on
    /// a step. A change beyond both bracket widths plus this margin is a step. A step smaller than that goes
    /// undetected, so the margin is also charged to every conversion as uncertainty (`step_tolerance_ns`).
    int64_t step_margin_ns = 100'000;
};

struct LocalClockObservation
{
    LocalClockSample sample;
    uint64_t generation = 0; ///< Changes whenever the realtime clock stepped, so older mappings are not reused.
    bool usable = false; ///< False when the bracket was too wide or the readings were invalid.
    int64_t step_tolerance_ns = 0; ///< Largest step this tracker can miss; added to the conversion's uncertainty.
    bool stepped = false; ///< True for the sample that revealed a step.
};

/// Watches successive samples of one host and numbers the intervals between detected realtime steps.
class LocalClockTracker
{
public:
    explicit LocalClockTracker(LocalClockConfig config = {});

    LocalClockObservation observe(const LocalClockSample& sample);

private:
    LocalClockConfig config_;
    bool have_previous_ = false;
    LocalClockSample previous_{};
    uint64_t generation_ = 1;
};

} // namespace isaaccapture::timesync
