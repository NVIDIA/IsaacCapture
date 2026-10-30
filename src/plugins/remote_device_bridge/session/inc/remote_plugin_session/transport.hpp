// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace remote_plugin_session
{

enum class ReceiveResult
{
    Ok,
    Timeout,
    Closed,
    Malformed,
};

/*!
 * @brief Message transport between a RemotePluginSession and the bridge.
 *
 * A frame is one complete wire message (24-byte header + payload) as built by
 * protocol::encode_frame. send() and receive() may run concurrently on two
 * threads; shutdown() unblocks both. Framing (TCP is a stream) lives in the
 * implementation, not in the callers.
 */
class ITransport
{
public:
    virtual ~ITransport() = default;

    //! False once the peer is gone or a send stalled past the transport's timeout.
    virtual bool send(const uint8_t* frame, size_t size) = 0;
    //! Timeout covers the wait for the first byte; Ok means `frame` holds exactly one complete message.
    virtual ReceiveResult receive(std::vector<uint8_t>& frame, std::chrono::milliseconds timeout) = 0;
    virtual void shutdown() = 0;
    //! Server side: finishes the transport's own handshake so send() works without waiting for a frame. Ok at once
    //! when there is none (TCP).
    virtual ReceiveResult wait_ready(std::chrono::milliseconds /*timeout*/)
    {
        return ReceiveResult::Ok;
    }
};

} // namespace remote_plugin_session
