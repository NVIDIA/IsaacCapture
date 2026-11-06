// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tcp_receiver.hpp"

#include <video_stage/clock_exchange.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

namespace video_stage
{

/// The Orin-to-PC mapping and the epoch it belongs to. The epoch increases whenever the Orin's clock was found to have
/// jumped relative to this one; frames mapped under different epochs must not be part of one synchronized take.
struct UpstreamMapping
{
    ClockMapping mapping;
    uint64_t epoch = 0;
};

/**
 * @brief The PC's half of the Orin clock exchange, over the control connection.
 *
 * Once a second it sends a four-timestamp request (the same wire the recorder and the browser use over LiveKit) and
 * feeds each reply to a ClockEstimator, so the Orin's monotonic stamps can be placed on this machine's monotonic clock.
 * A reader thread stamps t4 the instant a reply's first byte arrives. On a discontinuity the estimator restarts from
 * the new exchange; frames mapped in the interval before it was noticed carry the old offset.
 */
class ControlChannel
{
public:
    /// `first_epoch` differs for every session of a stream, so an epoch is never reused across a sender restart.
    ControlChannel(TcpReceiver& link, uint64_t first_epoch);
    ~ControlChannel();

    ControlChannel(const ControlChannel&) = delete;
    ControlChannel& operator=(const ControlChannel&) = delete;

    /// The current Orin-to-PC mapping with its uncertainty aged to now; nullopt before the first reply or when stale.
    std::optional<UpstreamMapping> mapping() const;

    uint64_t exchanges() const
    {
        return m_exchanges.load();
    }
    uint64_t rejected() const
    {
        return m_rejected.load();
    }
    uint64_t discontinuities() const
    {
        return m_discontinuities.load();
    }

private:
    void request_loop();
    void read_loop();

    TcpReceiver& m_link;
    mutable std::mutex m_mutex;
    ClockEstimator m_estimator;
    std::atomic<uint64_t> m_exchanges{ 0 };
    std::atomic<uint64_t> m_rejected{ 0 };
    std::atomic<uint64_t> m_discontinuities{ 0 };
    // Requests sent and not yet answered (sequence, t1): a reply is used only if it answers one of these.
    struct Outstanding
    {
        uint32_t sequence = 0;
        int64_t t1 = 0;
        bool used = true;
    };
    static constexpr size_t kOutstanding = 4;
    Outstanding m_outstanding[kOutstanding]{};
    size_t m_next_outstanding = 0;
    uint64_t m_epoch;
    std::mutex m_wait_mutex;
    std::condition_variable m_wait;
    bool m_stop = false;
    std::thread m_requester;
    std::thread m_reader;
};

} // namespace video_stage
