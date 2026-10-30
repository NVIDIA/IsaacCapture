// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/clock_offset_estimator.hpp"

#include <algorithm>
#include <stdexcept>

namespace remote_plugin_session
{

ClockOffsetEstimator::ClockOffsetEstimator(size_t max_samples, int64_t window_ns, size_t two_way_window)
    : m_max_samples(max_samples), m_window_ns(window_ns), m_two_way_window(two_way_window)
{
    if (max_samples == 0 || window_ns <= 0 || two_way_window == 0)
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
    while (!m_samples.empty() && m_samples.front().local_t_recv_ns < local_t_recv_ns - m_window_ns)
    {
        m_samples.pop_front();
    }

    m_offset_ns = std::min_element(m_samples.begin(), m_samples.end(),
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
    // local - remote, the same sign as the one-way offset.
    const int64_t offset_ns = ((t1_ns - t2_ns) + (t4_ns - t3_ns)) / 2;
    m_exchanges.push_back(Exchange{ rtt_ns, offset_ns });
    if (m_exchanges.size() > m_two_way_window)
    {
        m_exchanges.pop_front();
    }
}

const ClockOffsetEstimator::Exchange& ClockOffsetEstimator::best_exchange() const
{
    return *std::min_element(m_exchanges.begin(), m_exchanges.end(),
                             [](const Exchange& a, const Exchange& b) { return a.rtt_ns < b.rtt_ns; });
}

int64_t ClockOffsetEstimator::offset_ns() const
{
    return m_exchanges.empty() ? m_offset_ns : best_exchange().offset_ns;
}

int64_t ClockOffsetEstimator::remote_to_local(int64_t remote_ns) const
{
    return remote_ns + offset_ns();
}

int64_t ClockOffsetEstimator::rtt_min_ns() const
{
    return m_exchanges.empty() ? 0 : best_exchange().rtt_ns;
}

int64_t ClockOffsetEstimator::rtt_last_ns() const
{
    return m_exchanges.empty() ? 0 : m_exchanges.back().rtt_ns;
}

const char* ClockOffsetEstimator::offset_source() const
{
    return m_exchanges.empty() ? "one-way" : "two-way";
}

PingScheduler::PingScheduler(int64_t start_ns) : m_start_ns(start_ns), m_next_ns(start_ns)
{
}

PingScheduler::Action PingScheduler::poll(int64_t now_ns)
{
    if (m_state == State::GaveUp)
    {
        return Action::None;
    }
    if (m_state == State::Waiting && now_ns - m_start_ns >= kPongTimeoutNs)
    {
        m_state = State::GaveUp;
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
    if (m_state == State::Waiting)
    {
        m_state = State::GotPong;
    }
}

bool PingScheduler::active() const
{
    return m_state != State::GaveUp;
}

} // namespace remote_plugin_session
