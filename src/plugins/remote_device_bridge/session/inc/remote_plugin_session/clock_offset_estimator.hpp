// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

namespace remote_plugin_session
{

/*!
 * @brief Maps a remote monotonic clock onto the local one (local ~= remote + offset).
 *
 * One-way: fed with (remote send time, local receive time) of every received
 * frame; the offset is the running minimum of (local - remote) over a sliding
 * window, which hides one minimum one-way delay. Two-way: once a PING/PONG
 * exchange arrives the offset comes from the minimum-RTT exchange of the last
 * kTwoWayWindow instead (NTP filter; error = path asymmetry / 2) and one-way
 * samples no longer move it. Not thread-safe; the owner serializes access.
 */
class ClockOffsetEstimator
{
public:
    static constexpr size_t kTwoWayWindow = 32;

    explicit ClockOffsetEstimator(size_t max_samples = 200, int64_t window_ns = 5'000'000'000);

    void add_sample(int64_t remote_t_send_ns, int64_t local_t_recv_ns);
    //! t1/t4: local PING send / PONG receive; t2/t3: remote PING receive / PONG send. Negative RTTs are ignored.
    void add_two_way_sample(int64_t t1_ns, int64_t t2_ns, int64_t t3_ns, int64_t t4_ns);

    //! Zero until the first sample arrives.
    int64_t offset_ns() const;
    int64_t remote_to_local(int64_t remote_ns) const;
    //! Zero until the first two-way exchange.
    int64_t rtt_min_ns() const;
    int64_t rtt_last_ns() const;
    //! "two-way" once an exchange was added, else "one-way".
    const char* offset_source() const;

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
    int64_t m_offset_ns = 0;
    std::deque<Exchange> m_exchanges;
    int64_t m_two_way_offset_ns = 0;
    int64_t m_rtt_min_ns = 0;
    int64_t m_rtt_last_ns = 0;
};

/*!
 * @brief When the bridge sends PINGs: a burst right after HELLO_ACK, then 1 Hz.
 *
 * Without any PONG within kPongTimeout of the start, poll() returns GiveUp once
 * and nothing after that: the peer stays on the one-way estimate.
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
    //! False after GiveUp; late PONGs are then ignored.
    bool active() const;

private:
    int64_t m_start_ns;
    int64_t m_next_ns;
    int m_sent = 0;
    bool m_got_pong = false;
    bool m_gave_up = false;
};

//! True when a mapped sample time is older than now - max_age; max_age <= 0 disables the guard. After a link
//! stall the backlog arrives in a burst; pushing it into OpenXR would replay old motion instead of skipping it.
inline bool should_drop_stale(int64_t mapped_sample_ns, int64_t now_ns, int64_t max_age_ns)
{
    return max_age_ns > 0 && now_ns - mapped_sample_ns > max_age_ns;
}

} // namespace remote_plugin_session
