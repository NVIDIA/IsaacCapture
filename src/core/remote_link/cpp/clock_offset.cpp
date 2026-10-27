// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_link/clock_offset.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace core
{

ClockOffsetEstimator::ClockOffsetEstimator(size_t max_samples, int64_t window_ns)
    : m_max_samples(max_samples), m_window_ns(window_ns)
{
    if (max_samples == 0 || window_ns <= 0)
    {
        throw std::invalid_argument("ClockOffsetEstimator requires a non-empty window");
    }
}

void ClockOffsetEstimator::add_sample(int64_t remote_t_send_ns, int64_t local_t_recv_ns)
{
    m_samples.push_back(Sample{ local_t_recv_ns, local_t_recv_ns - remote_t_send_ns });
    while (m_samples.size() > m_max_samples)
    {
        m_samples.pop_front();
    }
    while (m_samples.front().local_t_recv_ns < local_t_recv_ns - m_window_ns)
    {
        m_samples.pop_front();
    }
    m_one_way_offset_ns = std::min_element(m_samples.begin(), m_samples.end(),
                                           [](const Sample& a, const Sample& b) { return a.offset_ns < b.offset_ns; })
                              ->offset_ns;
}

void ClockOffsetEstimator::add_two_way_sample(int64_t t1_ns, int64_t t2_ns, int64_t t3_ns, int64_t t4_ns)
{
    const int64_t rtt_ns = (t4_ns - t1_ns) - (t3_ns - t2_ns);
    if (rtt_ns < 0)
    {
        return;
    }
    m_exchanges.push_back(Exchange{ rtt_ns, ((t1_ns - t2_ns) + (t4_ns - t3_ns)) / 2 });
    if (m_exchanges.size() > kTwoWayWindow)
    {
        m_exchanges.pop_front();
    }
    const Exchange& best = *std::min_element(m_exchanges.begin(), m_exchanges.end(),
                                             [](const Exchange& a, const Exchange& b) { return a.rtt_ns < b.rtt_ns; });
    m_two_way_offset_ns = best.offset_ns;
    m_rtt_min_ns = best.rtt_ns;
    m_rtt_last_ns = rtt_ns;
}

int64_t ClockOffsetEstimator::offset_ns() const
{
    return m_exchanges.empty() ? m_one_way_offset_ns : m_two_way_offset_ns;
}

int64_t ClockOffsetEstimator::remote_to_local(int64_t remote_ns) const
{
    return remote_ns + offset_ns();
}

int64_t ClockOffsetEstimator::rtt_min_ns() const
{
    return m_rtt_min_ns;
}

int64_t ClockOffsetEstimator::rtt_last_ns() const
{
    return m_rtt_last_ns;
}

bool ClockOffsetEstimator::has_sample() const
{
    return !m_samples.empty() || !m_exchanges.empty();
}

bool ClockOffsetEstimator::two_way() const
{
    return !m_exchanges.empty();
}

PingScheduler::PingScheduler(int64_t start_ns) : m_start_ns(start_ns), m_next_ns(start_ns)
{
}

PingScheduler::Action PingScheduler::poll(int64_t now_ns)
{
    if (m_gave_up)
    {
        return Action::None;
    }
    if (!m_got_pong && now_ns - m_start_ns >= kPongTimeoutNs)
    {
        m_gave_up = true;
        return Action::GiveUp;
    }
    if (now_ns < m_next_ns)
    {
        return Action::None;
    }
    ++m_sent;
    m_next_ns = now_ns + (m_sent < kBurstCount ? kBurstPeriodNs : kSteadyPeriodNs);
    return Action::Ping;
}

void PingScheduler::on_pong()
{
    m_got_pong = true;
}

bool PingScheduler::active() const
{
    return !m_gave_up;
}

int64_t PingScheduler::next_due_ns() const
{
    if (m_gave_up)
    {
        return std::numeric_limits<int64_t>::max();
    }
    return m_got_pong ? m_next_ns : std::min(m_next_ns, m_start_ns + kPongTimeoutNs);
}

} // namespace core
