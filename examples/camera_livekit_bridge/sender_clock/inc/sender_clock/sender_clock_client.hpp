// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <clock_exchange/clock_estimator.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace camera_bridge
{

struct SenderClockConfig
{
    /// IPv4 address of the sender (a literal address, not a name) and its clock responder's UDP port.
    std::string host;
    std::uint16_t port = 0;
    /// Time between requests. A step of the sender's clock is seen at the next reply, so this bounds how long a
    /// stale mapping can be used.
    std::chrono::milliseconds interval{ 200 };
    /// How long a reply is waited for. Later replies are discarded: a measurement needs t4 near the receipt.
    std::chrono::milliseconds reply_timeout{ 150 };
};

struct SenderClockSnapshot
{
    std::optional<ClockMapping> mapping;
    bool discontinuity = false;
    uint64_t epoch = 0;
    std::size_t accepted_in_epoch = 0;
    uint64_t exchanges = 0;
    uint64_t rejected = 0;
    /// Replies that were not a measurement: wrong size or tag, wrong peer, not matching the outstanding request, or
    /// from before a reset.
    uint64_t ignored = 0;
    uint64_t timeouts = 0;
    /// Bumped by every reset; an exchange only counts if its request was sent in the current generation.
    uint64_t generation = 0;
    /// Why the last request failed, empty while healthy.
    std::string last_error;
};

/**
 * @brief Measures the sender's clock against this host's CLOCK_MONOTONIC over the sender's UDP clock responder.
 *
 * A thread sends the 13-byte request ('Q', u32 seq, i64 t1, little-endian) to the sender at the configured interval
 * and feeds the 29-byte reply ('R', seq, t1, t2, t3: the sender's clock at receipt and at send) to a
 * ClockEstimator. Only a reply from the sender's address, of exactly that size and tag, matching the one
 * outstanding request, and sent in the current generation counts. A loss is a missed sample; there is no
 * connection to lose, so a reset (new session) is the only thing that invalidates earlier measurements.
 */
class SenderClockClient
{
public:
    explicit SenderClockClient(SenderClockConfig config);
    ~SenderClockClient();

    SenderClockClient(const SenderClockClient&) = delete;
    SenderClockClient& operator=(const SenderClockClient&) = delete;

    void start();
    void stop();

    /// Start a new mapping epoch: nothing measured before applies, and the outstanding request is void.
    void reset();

    /// Tell the client which capture session the frames now belong to. The first call and every change of session id
    /// reset it (once, however many streams report it).
    void begin_session(uint64_t session_id);

    /// A consistent copy of the estimator's state, with the mapping's uncertainty aged to now.
    SenderClockSnapshot snapshot() const;

private:
    void run();

    SenderClockConfig m_config;
    mutable std::mutex m_mutex;
    ClockEstimator m_estimator;
    uint64_t m_generation = 0;
    uint64_t m_exchanges = 0;
    uint64_t m_rejected = 0;
    uint64_t m_ignored = 0;
    uint64_t m_timeouts = 0;
    std::string m_last_error;
    std::optional<uint64_t> m_session_id;
    std::atomic<bool> m_running{ false };
    std::thread m_thread;
};

} // namespace camera_bridge
