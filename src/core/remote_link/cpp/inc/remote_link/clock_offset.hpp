// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

namespace core
{

/*!
 * @brief Maps the peer's monotonic clock onto the local one: offset = local - remote.
 *
 * One-way: the minimum (local receive - remote send) over a sliding window, biased by the minimum one-way delay.
 * Two-way: once a PING/PONG exchange arrives, the offset of the minimum-RTT exchange of the last kTwoWayWindow
 * (error = path asymmetry / 2); one-way samples no longer move it. Not thread-safe.
 */
class ClockOffsetEstimator
{
public:
    static constexpr size_t kTwoWayWindow = 32;

    explicit ClockOffsetEstimator(size_t max_samples = 200, int64_t window_ns = 5'000'000'000);

    void add_sample(int64_t remote_t_send_ns, int64_t local_t_recv_ns);
    //! t1/t4: local PING send / PONG receive; t2/t3: remote PING receive / PONG send. Negative RTTs are ignored.
    void add_two_way_sample(int64_t t1_ns, int64_t t2_ns, int64_t t3_ns, int64_t t4_ns);

    //! Zero until the first sample.
    int64_t offset_ns() const;
    int64_t remote_to_local(int64_t remote_ns) const;
    //! Zero until the first two-way exchange.
    int64_t rtt_min_ns() const;
    int64_t rtt_last_ns() const;
    bool has_sample() const;
    bool two_way() const;

private:
    struct Sample
    {
        int64_t local_t_recv_ns;
        int64_t offset_ns;
    };
    struct Exchange
    {
        int64_t rtt_ns;
        int64_t offset_ns;
    };

    size_t m_max_samples;
    int64_t m_window_ns;
    std::deque<Sample> m_samples;
    int64_t m_one_way_offset_ns = 0;
    std::deque<Exchange> m_exchanges;
    int64_t m_two_way_offset_ns = 0;
    int64_t m_rtt_min_ns = 0;
    int64_t m_rtt_last_ns = 0;
};

/*!
 * @brief When to send PINGs: a burst of kBurstCount every kBurstPeriodNs from the start, then 1 Hz.
 *
 * Without any PONG within kPongTimeoutNs of the start, poll() returns GiveUp once and nothing after that.
 * Not thread-safe.
 */
class PingScheduler
{
public:
    enum class Action
    {
        None,
        Ping,
        GiveUp,
    };

    static constexpr int kBurstCount = 10;
    static constexpr int64_t kBurstPeriodNs = 50'000'000;
    static constexpr int64_t kSteadyPeriodNs = 1'000'000'000;
    static constexpr int64_t kPongTimeoutNs = 2'000'000'000;

    explicit PingScheduler(int64_t start_ns);

    Action poll(int64_t now_ns);
    void on_pong();
    //! False after GiveUp.
    bool active() const;
    //! Earliest time poll() may return something other than None; INT64_MAX once inactive.
    int64_t next_due_ns() const;

private:
    int64_t m_start_ns;
    int64_t m_next_ns;
    int m_sent = 0;
    bool m_got_pong = false;
    bool m_gave_up = false;
};

} // namespace core
