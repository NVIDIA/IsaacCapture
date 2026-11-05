// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace plugins
{
namespace sensing
{

/// What a backend's `CaptureFrame::sensor_time` counts.
enum class SensorTimeDomain
{
    /// The backend reports no sensor time.
    None,
    /// SIPL `frameCaptureTSC`: ticks of the ARM generic counter (assumed; checked at run time by SensorClockMap).
    GenericCounterTicks,
    /// Argus `getSensorTimestamp()`: nanoseconds on CLOCK_MONOTONIC_RAW (assumed; checked at run time).
    MonotonicRawNs,
};

/// A source clock reading paired with CLOCK_MONOTONIC, read back to back.
struct ClockPair
{
    /// The source clock's value (counter ticks or raw ns) at the midpoint of the bracket.
    int64_t source = 0;
    /// CLOCK_MONOTONIC ns at the same midpoint.
    int64_t monotonic_ns = 0;
    /// Width of the bracket the pair was read in; the pair is no more certain than this.
    int64_t bracket_ns = 0;
};

/// Reads the ARM generic counter (CNTVCT_EL0) and CLOCK_MONOTONIC back to back, keeping the narrowest of a few
/// attempts. Throws where the counter is not readable (not aarch64).
ClockPair read_counter_pair();
/// The generic counter's frequency in Hz (CNTFRQ_EL0).
int64_t read_counter_hz();
/// Reads CLOCK_MONOTONIC_RAW and CLOCK_MONOTONIC back to back, keeping the narrowest of a few attempts.
ClockPair read_raw_pair();

/**
 * @brief Places a backend's sensor timestamp on this machine's CLOCK_MONOTONIC.
 *
 * One anchor pair (source clock, CLOCK_MONOTONIC) is read through a supported clock API and refreshed at a low
 * rate, never per frame; a frame's capture time is the anchor plus the source-clock difference. The source's units
 * are an assumption, not a fact (see SensorTimeDomain), so every conversion is checked: a capture time that is in
 * the future of the frame's ready time, or more than `max_age_ns` before it, means the assumed domain or rate is
 * wrong and the conversion throws. There is no fallback to a host-side stamp.
 *
 * Not thread-safe; one instance per capture thread.
 */
class SensorClockMap
{
public:
    /// Reads one pair; injectable so the mapping is testable off-target.
    using PairReader = std::function<ClockPair()>;

    struct Config
    {
        SensorTimeDomain domain = SensorTimeDomain::None;
        /// Counter frequency in Hz; required for GenericCounterTicks.
        int64_t counter_hz = 0;
        /// Anchor refresh period.
        int64_t refresh_ns = 1'000'000'000;
        /// Largest plausible gap between capture and the frame being ready.
        int64_t max_age_ns = 2'000'000'000;
    };

    SensorClockMap(Config config, PairReader reader) : m_config(config), m_reader(std::move(reader))
    {
        if (m_config.domain == SensorTimeDomain::None)
            throw std::invalid_argument("SensorClockMap needs a sensor time domain");
        if (m_config.domain == SensorTimeDomain::GenericCounterTicks && m_config.counter_hz <= 0)
            throw std::invalid_argument("SensorClockMap needs the counter frequency for counter ticks");
    }

    /**
     * @brief CLOCK_MONOTONIC ns of the sensor's capture event.
     * @param sensor_time The backend's raw sensor timestamp.
     * @param ready_monotonic_ns CLOCK_MONOTONIC when the frame was ready (the plausibility reference).
     * @param now_monotonic_ns CLOCK_MONOTONIC now, to decide when the anchor is due.
     * @throws std::runtime_error when the value is not plausible for the assumed domain.
     */
    int64_t to_monotonic_ns(uint64_t sensor_time, int64_t ready_monotonic_ns, int64_t now_monotonic_ns)
    {
        if (!m_have_anchor || now_monotonic_ns - m_anchor_taken_ns >= m_config.refresh_ns)
        {
            m_anchor = m_reader();
            m_have_anchor = true;
            m_anchor_taken_ns = now_monotonic_ns;
        }
        const int64_t delta = to_ns(static_cast<int64_t>(sensor_time) - m_anchor.source);
        const int64_t capture = m_anchor.monotonic_ns + delta;
        if (capture > ready_monotonic_ns || ready_monotonic_ns - capture > m_config.max_age_ns)
        {
            throw std::runtime_error(
                "sensor timestamp " + std::to_string(sensor_time) + " maps to " +
                std::to_string(capture - ready_monotonic_ns) +
                " ns relative to the frame being ready: the assumed sensor clock domain or rate is wrong");
        }
        return capture;
    }

    /// Uncertainty of the current anchor (its bracket); zero before the first conversion.
    int64_t anchor_uncertainty_ns() const
    {
        return m_have_anchor ? m_anchor.bracket_ns : 0;
    }

private:
    int64_t to_ns(int64_t source_delta) const
    {
        if (m_config.domain == SensorTimeDomain::MonotonicRawNs)
            return source_delta;
        // Split into whole seconds and remainder so the product cannot overflow 64 bits.
        const int64_t hz = m_config.counter_hz;
        const int64_t secs = source_delta / hz;
        const int64_t rem = source_delta % hz;
        return secs * 1'000'000'000 + rem * 1'000'000'000 / hz;
    }

    Config m_config;
    PairReader m_reader;
    ClockPair m_anchor{};
    int64_t m_anchor_taken_ns = 0;
    bool m_have_anchor = false;
};

} // namespace sensing
} // namespace plugins
