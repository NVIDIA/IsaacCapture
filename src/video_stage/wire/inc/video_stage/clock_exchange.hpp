// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>

namespace video_stage
{

/// One completed four-timestamp exchange: t1/t4 on the local clock, t2/t3 on the remote's. Nanoseconds.
struct Exchange
{
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t t3 = 0;
    int64_t t4 = 0;

    int64_t rtt_ns() const
    {
        return (t4 - t1) - (t3 - t2);
    }
    /// Remote minus local, assuming symmetric transit.
    int64_t offset_ns() const
    {
        return ((t2 - t1) + (t3 - t4)) / 2;
    }
    bool well_formed() const
    {
        return t4 >= t1 && t3 >= t2 && rtt_ns() >= 0;
    }
};

struct ClockMapping
{
    int64_t offset_ns = 0; ///< remote minus local
    int64_t rtt_ns = 0;
    /// Half the round trip plus the drift the exchange's age allows: what unknown asymmetry and rate can cause.
    int64_t uncertainty_ns = 0;
    int64_t accepted_at_ns = 0;

    int64_t to_local_ns(int64_t remote_ns) const
    {
        return remote_ns - offset_ns;
    }
};

/**
 * @brief Maps a remote monotonic clock onto this one from four-timestamp exchanges.
 *
 * Same rules as the Python estimator the recorder uses (egocap_web/clock_exchange.py): the best recent exchange by
 * uncertainty, a mapping withheld when no exchange arrived within the stale limit, a new exchange that disagrees with
 * the current mapping by more than both uncertainties plus a drift allowance flagged as a discontinuity, and the
 * uncertainty aged to the time of the read. Not thread-safe; the owner serialises access.
 */
class ClockEstimator
{
public:
    static constexpr int64_t kMaxRttNs = 250'000'000;
    static constexpr int64_t kStaleAfterNs = 5'000'000'000;
    static constexpr size_t kWindow = 8;
    static constexpr double kDriftAllowance = 200e-6;

    /// Why the exchange was rejected, or empty when accepted.
    std::string add(const Exchange& e)
    {
        if (!e.well_formed())
            return "malformed exchange";
        if (e.rtt_ns() > kMaxRttNs)
            return "round trip too long";
        const std::optional<ClockMapping> previous = m_current;
        m_newest_ns = e.t4;
        const ClockMapping newest = mapping_of(e, e.t4);
        if (previous && disagrees(*previous, newest))
        {
            m_discontinuity = true;
            m_recent.clear();
            m_recent.push_back(e);
            m_current = mapping_of(e, e.t4);
        }
        else
        {
            m_recent.push_back(e);
            if (m_recent.size() > kWindow)
                m_recent.pop_front();
            const Exchange* best = &m_recent.front();
            for (const Exchange& c : m_recent)
            {
                const int64_t u = uncertainty(c, e.t4);
                const int64_t bu = uncertainty(*best, e.t4);
                if (u < bu || (u == bu && c.t4 > best->t4))
                    best = &c;
            }
            m_current = mapping_of(*best, e.t4);
        }
        return {};
    }

    /// The mapping at `now_ns` with its uncertainty aged to then, or nullopt when there is none or it is stale.
    std::optional<ClockMapping> mapping(int64_t now_ns) const
    {
        if (!m_current || !m_newest_ns || now_ns - *m_newest_ns > kStaleAfterNs)
            return std::nullopt;
        ClockMapping aged = *m_current;
        const int64_t age = std::max<int64_t>(0, now_ns - aged.accepted_at_ns);
        aged.uncertainty_ns =
            std::max(aged.uncertainty_ns, aged.rtt_ns / 2 + static_cast<int64_t>(kDriftAllowance * age));
        return aged;
    }

    /// Set when the clocks jumped apart; frames already mapped keep their mapping. The owner counts the event and
    /// clears it; the estimator has already restarted from the exchange that showed it.
    bool discontinuity() const
    {
        return m_discontinuity;
    }
    void clear_discontinuity()
    {
        m_discontinuity = false;
    }

private:
    static int64_t uncertainty(const Exchange& e, int64_t now_ns)
    {
        return e.rtt_ns() / 2 + static_cast<int64_t>(kDriftAllowance * std::max<int64_t>(0, now_ns - e.t4));
    }

    static ClockMapping mapping_of(const Exchange& e, int64_t now_ns)
    {
        return ClockMapping{ e.offset_ns(), e.rtt_ns(), uncertainty(e, now_ns), e.t4 };
    }

    static bool disagrees(const ClockMapping& previous, const ClockMapping& candidate)
    {
        const int64_t elapsed = std::max<int64_t>(0, candidate.accepted_at_ns - previous.accepted_at_ns);
        const double allowed = static_cast<double>(previous.uncertainty_ns + candidate.uncertainty_ns) +
                               kDriftAllowance * static_cast<double>(elapsed);
        const int64_t delta = candidate.offset_ns - previous.offset_ns;
        return static_cast<double>(delta < 0 ? -delta : delta) > allowed;
    }

    std::deque<Exchange> m_recent;
    std::optional<ClockMapping> m_current;
    std::optional<int64_t> m_newest_ns;
    bool m_discontinuity = false;
};

} // namespace video_stage
