// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <remote_link/message_header.hpp>
#include <remote_link/tcp_transport.hpp>
#include <sys/socket.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

namespace remote_link_test
{

constexpr std::chrono::milliseconds kWait{ 2000 };
constexpr std::chrono::milliseconds kShort{ 200 };

inline std::pair<core::TcpSocket, core::TcpSocket> socket_pair()
{
    int fds[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
    return { core::TcpSocket(fds[0]), core::TcpSocket(fds[1]) };
}

//! A link message of exactly `message_size` bytes (header included) with a patterned payload.
inline std::vector<uint8_t> message_of_size(size_t message_size, uint8_t type = 34)
{
    std::vector<uint8_t> payload(message_size - core::kHeaderSize);
    for (size_t i = 0; i < payload.size(); ++i)
    {
        payload[i] = static_cast<uint8_t>(i * 31 + 7);
    }
    return core::encode_link_message(type, payload);
}

inline bool wait_until(const std::function<bool()>& predicate, std::chrono::milliseconds timeout = kWait)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate())
    {
        if (std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return true;
}

} // namespace remote_link_test
