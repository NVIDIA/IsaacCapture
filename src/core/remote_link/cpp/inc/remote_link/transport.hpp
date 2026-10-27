// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace core
{

//! The peer closed or reset the connection before it sent any message or part of one, e.g. a server that crashed or
//! was killed before it took the connection up. Connecting again may succeed.
class PeerClosedError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

enum class ReceiveResult
{
    Ok,
    Timeout,
    Closed, //!< the connection ended between messages
    Malformed,
    Truncated, //!< the connection ended or stalled partway through a message
};

/*!
 * @brief Moves complete link messages (link message header + payload) between two peers.
 *
 * send() and receive() may run concurrently on two threads; shutdown() unblocks both from a third.
 * Framing lives in the implementation.
 */
class ITransport
{
public:
    virtual ~ITransport() = default;

    //! Sends one complete message; false means the link is unusable.
    virtual bool send(const uint8_t* message, size_t size) = 0;
    //! Ok: `message` holds exactly one complete message. The timeout covers the wait for its first byte.
    virtual ReceiveResult receive(std::vector<uint8_t>& message, std::chrono::milliseconds first_byte_timeout) = 0;
    virtual void shutdown() = 0;
    //! Ends the sending direction after the last send: the peer reads everything sent, then end of stream.
    //! receive() keeps working. Call it only when no send() is in progress or will follow.
    virtual void shutdown_write()
    {
    }
    //! Completes the transport's own handshake (server WebSocket upgrade); Ok at once when there is none.
    virtual ReceiveResult wait_ready(std::chrono::milliseconds /*timeout*/)
    {
        return ReceiveResult::Ok;
    }
    //! True when a send would not queue behind unsent bytes.
    virtual bool wait_writable(std::chrono::milliseconds /*timeout*/)
    {
        return true;
    }
};

} // namespace core
