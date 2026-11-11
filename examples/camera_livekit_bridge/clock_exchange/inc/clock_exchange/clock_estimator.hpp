// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>

namespace camera_bridge
{

/// Round-trip budget, staleness limit, window and relative clock-rate allowance of the estimator; the same values
/// and the same decisions as the four-timestamp estimator egocap applies to the clock of a camera publisher.
inline constexpr int64_t kDefaultMaxRttNs = 250'000'000;
inline constexpr int64_t kDefaultStaleAfterNs = 5'000'000'000;
inline constexpr std::size_t kDefaultWindow = 8;
inline constexpr double kDefaultDriftAllowance = 200e-6;

/// Floor division, so a negative offset rounds the same way in C++ as in the integer arithmetic of the reference.
inline int64_t floor_div(int64_t numerator, int64_t denominator)
{
    int64_t quotient = numerator / denominator;
    if ((numerator % denominator != 0) && ((numerator < 0) != (denominator < 0)))
        --quotient;
    return quotient;
}

/// One completed exchange: t1 and t4 on the local clock, t2 and t3 on the remote clock.
struct Exchange
{
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t t3 = 0;
    int64_t t4 = 0;

    /// Round trip with the remote's own processing time removed.
    int64_t rtt_ns() const
    {
        return (t4 - t1) - (t3 - t2);
    }
    /// Remote minus local, assuming the two directions took equally long.
    int64_t offset_ns() const
    {
        return floor_div((t2 - t1) + (t3 - t4), 2);
    }
    /// Time moved forward on both clocks and the round trip is not negative.
    bool well_formed() const
    {
        return t4 >= t1 && t3 >= t2 && rtt_ns() >= 0;
    }
};

/// A mapping of the remote clock onto the local one. The uncertainty is half the round trip plus the drift the age
/// allows; it is an error bound for asymmetric paths and clock rates, not an accuracy claim about anything else.
struct ClockMapping
{
    uint64_t epoch = 0;
    int64_t offset_ns = 0;
    int64_t rtt_ns = 0;
    int64_t uncertainty_ns = 0;
    /// Local time at which the exchange behind this mapping completed.
    int64_t accepted_at_ns = 0;

    int64_t to_local_ns(int64_t remote_ns) const
    {
        return remote_ns - offset_ns;
    }
    bool operator==(const ClockMapping& other) const
    {
        return epoch == other.epoch && offset_ns == other.offset_ns && rtt_ns == other.rtt_ns &&
               uncertainty_ns == other.uncertainty_ns && accepted_at_ns == other.accepted_at_ns;
    }
};

/**
 * @brief Keeps the best recent exchange as the current mapping and notices when the clocks jump apart.
 *
 * Not thread-safe: one owner calls add(), reset() and mapping().
 */
class ClockEstimator
{
public:
    explicit ClockEstimator(int64_t max_rtt_ns = kDefaultMaxRttNs,
                            int64_t stale_after_ns = kDefaultStaleAfterNs,
                            std::size_t window = kDefaultWindow,
                            double drift_allowance = kDefaultDriftAllowance)
        : m_max_rtt_ns(max_rtt_ns), m_stale_after_ns(stale_after_ns), m_window(window), m_drift_allowance(drift_allowance)
    {
    }

    /// Start over, for a new source session or a restarted remote: the old mapping no longer applies.
    void reset()
    {
        m_recent.clear();
        m_current.reset();
        m_newest_ns.reset();
        m_discontinuity = false;
        m_accepted_in_epoch = 0;
        ++m_epoch;
    }

    /// Takes one exchange into account; returns why it was rejected, or nothing when it was accepted.
    std::optional<std::string> add(const Exchange& exchange)
    {
        if (!exchange.well_formed())
            return "malformed exchange";
        if (exchange.rtt_ns() > m_max_rtt_ns)
            return "round trip too long";
        const std::optional<ClockMapping> previous = m_current;
        m_newest_ns = exchange.t4;
        const ClockMapping newest = mapping_of(exchange, exchange.t4);
        ClockMapping candidate;
        if (previous && disagrees(*previous, newest))
        {
            // Not a refinement: the clocks moved. Judged on the newest exchange alone, so a jump is seen at once.
            m_discontinuity = true;
            ++m_epoch;
            m_accepted_in_epoch = 0;
            m_recent.clear();
            m_recent.push_back(exchange);
            candidate = mapping_of(exchange, exchange.t4);
        }
        else
        {
            m_recent.push_back(exchange);
            while (m_recent.size() > m_window)
                m_recent.pop_front();
            const Exchange* best = &m_recent.front();
            for (const Exchange& e : m_recent)
            {
                const int64_t ue = uncertainty_ns(e, exchange.t4);
                const int64_t ub = uncertainty_ns(*best, exchange.t4);
                if (ue < ub || (ue == ub && e.t4 > best->t4))
                    best = &e;
            }
            candidate = mapping_of(*best, exchange.t4);
        }
        ++m_accepted_in_epoch;
        m_current = candidate;
        return std::nullopt;
    }

    /// The current mapping with its uncertainty aged to `now_ns`, or nothing when there is none yet or no exchange
    /// was accepted within the staleness limit.
    std::optional<ClockMapping> mapping(int64_t now_ns) const
    {
        if (!m_current || !m_newest_ns || now_ns - *m_newest_ns > m_stale_after_ns)
            return std::nullopt;
        ClockMapping aged = *m_current;
        const int64_t age = std::max<int64_t>(0, now_ns - aged.accepted_at_ns);
        const int64_t wanted = aged.rtt_ns / 2 + static_cast<int64_t>(m_drift_allowance * static_cast<double>(age));
        if (wanted > aged.uncertainty_ns)
            aged.uncertainty_ns = wanted;
        return aged;
    }

    /// Set when a new best mapping disagreed with the last by more than their uncertainties and the allowed drift
    /// explain, and left set until reset().
    bool discontinuity() const
    {
        return m_discontinuity;
    }
    uint64_t epoch() const
    {
        return m_epoch;
    }
    /// Exchanges accepted since the epoch began (a new epoch starts at reset() and at a discontinuity).
    std::size_t accepted_in_epoch() const
    {
        return m_accepted_in_epoch;
    }

private:
    int64_t uncertainty_ns(const Exchange& e, int64_t now_ns) const
    {
        return e.rtt_ns() / 2 +
               static_cast<int64_t>(m_drift_allowance * static_cast<double>(std::max<int64_t>(0, now_ns - e.t4)));
    }

    ClockMapping mapping_of(const Exchange& e, int64_t now_ns) const
    {
        ClockMapping m;
        m.epoch = m_epoch;
        m.offset_ns = e.offset_ns();
        m.rtt_ns = e.rtt_ns();
        m.uncertainty_ns = uncertainty_ns(e, now_ns);
        m.accepted_at_ns = e.t4;
        return m;
    }

    bool disagrees(const ClockMapping& previous, const ClockMapping& candidate) const
    {
        const int64_t elapsed = std::max<int64_t>(0, candidate.accepted_at_ns - previous.accepted_at_ns);
        const double allowed = static_cast<double>(previous.uncertainty_ns) +
                               static_cast<double>(candidate.uncertainty_ns) +
                               m_drift_allowance * static_cast<double>(elapsed);
        const int64_t delta = candidate.offset_ns - previous.offset_ns;
        return static_cast<double>(delta < 0 ? -delta : delta) > allowed;
    }

    int64_t m_max_rtt_ns;
    int64_t m_stale_after_ns;
    std::size_t m_window;
    double m_drift_allowance;
    std::deque<Exchange> m_recent;
    uint64_t m_epoch = 1;
    std::optional<ClockMapping> m_current;
    std::optional<int64_t> m_newest_ns;
    bool m_discontinuity = false;
    std::size_t m_accepted_in_epoch = 0;
};

} // namespace camera_bridge
