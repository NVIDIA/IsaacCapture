// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_link/reject.hpp"

#include "inc/remote_link/message_header.hpp"
#include "inc/remote_link/messages.hpp"
#include "inc/remote_link/transport.hpp"
#include "inc/remote_link/websocket_transport.hpp"

#include <oxr_utils/os_time.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace core
{

namespace
{

constexpr std::chrono::milliseconds kStep{ 1000 };

} // namespace

void reject_remote_connection(int fd, bool websocket, const std::string& text)
{
    const std::unique_ptr<ITransport> transport = adopt_transport(fd, websocket);
    if (transport->wait_ready(kStep) != ReceiveResult::Ok)
    {
        return;
    }
    std::vector<uint8_t> message;
    const ReceiveResult hello = transport->receive(message, kStep);
    if (hello != ReceiveResult::Ok && hello != ReceiveResult::Timeout)
    {
        return;
    }

    std::vector<uint8_t> error = encode_link_message(
        static_cast<uint8_t>(LinkMessageType::Error), std::vector<uint8_t>(text.begin(), text.end()));
    stamp_header(error.data(), 0, os_monotonic_now_ns());
    if (!transport->send(error.data(), error.size()))
    {
        return;
    }
    transport->shutdown_write();

    const auto deadline = std::chrono::steady_clock::now() + kStep;
    while (true)
    {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (remaining.count() <= 0)
        {
            return;
        }
        const ReceiveResult result = transport->receive(message, remaining);
        if (result != ReceiveResult::Ok && result != ReceiveResult::Timeout)
        {
            return;
        }
    }
}

} // namespace core
