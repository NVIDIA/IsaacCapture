/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CLOCK_RESPONDER_H
#define ARGUS_SENDER_CLOCK_RESPONDER_H

// Answers the clock exchange (see ClockExchange.h) over UDP on the sender's CLOCK_MONOTONIC, for as long as the object
// lives. It is stateless and unauthenticated, like the AVS1 stream it sits beside, and exposes nothing but the sender's
// own clock readings. Anything that is not exactly one request is ignored. The requester matches replies to its
// outstanding (seq, t1), so a late reply from before a restart is rejected by the client.

#include <cstdint>
#include <memory>
#include <string>

namespace argus_sender
{

struct ClockResponderConfig
{
    std::string bindHost = "0.0.0.0";
    // 0 picks a free port; read it back with port().
    uint16_t port = 0;
};

class ClockResponder
{
public:
    // Starts answering. Throws std::runtime_error with the reason if the address cannot be bound.
    explicit ClockResponder(const ClockResponderConfig& config);
    ~ClockResponder();

    ClockResponder(const ClockResponder&) = delete;
    ClockResponder& operator=(const ClockResponder&) = delete;

    uint16_t port() const;
    // Requests answered so far.
    uint64_t answered() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_CLOCK_RESPONDER_H
