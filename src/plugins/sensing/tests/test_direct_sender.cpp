// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Destroying a DirectSender must be bounded whatever its worker thread is blocked in: a connect that never
// completes, a send to a peer that does not read, or an idle connection.

#include "core/direct_sender.hpp"

#include <arpa/inet.h>
#include <catch2/catch_test_macros.hpp>
#include <netinet/in.h>
#include <sys/socket.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <poll.h>
#include <thread>
#include <unistd.h>
#include <vector>

using plugins::sensing::DirectSender;
using plugins::sensing::DirectSenderConfig;
using plugins::sensing::EncodedVideoFrame;

namespace
{

using Clock = std::chrono::steady_clock;
constexpr auto kBound = std::chrono::seconds(2);

class Listener
{
public:
    explicit Listener(int backlog)
    {
        m_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        REQUIRE(::bind(m_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        REQUIRE(::listen(m_fd, backlog) == 0);
        socklen_t len = sizeof(addr);
        ::getsockname(m_fd, reinterpret_cast<sockaddr*>(&addr), &len);
        m_port = ntohs(addr.sin_port);
    }

    ~Listener()
    {
        ::close(m_fd);
        for (const int fd : m_accepted)
            ::close(fd);
    }

    uint16_t port() const
    {
        return m_port;
    }

    /// Accepts \p count connections and reads their 4-byte hellos, then never reads again.
    bool accept_and_stall(int count)
    {
        for (int i = 0; i < count; ++i)
        {
            pollfd pfd{ m_fd, POLLIN, 0 };
            if (::poll(&pfd, 1, 5000) <= 0)
                return false;
            const int c = ::accept(m_fd, nullptr, nullptr);
            if (c < 0)
                return false;
            m_accepted.push_back(c);
            char hello[4];
            if (::recv(c, hello, sizeof(hello), MSG_WAITALL) != 4)
                return false;
        }
        return true;
    }

    /// Fills the accept queue with half-open clients so later SYNs are dropped and a connect hangs.
    void saturate(std::vector<int>& clients)
    {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(m_port);
        for (int i = 0; i < 8; ++i)
        {
            const int c = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
            ::connect(c, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
            clients.push_back(c);
        }
    }

private:
    int m_fd = -1;
    uint16_t m_port = 0;
    std::vector<int> m_accepted;
};

DirectSenderConfig config_for(uint16_t port)
{
    DirectSenderConfig config;
    config.host = "127.0.0.1";
    config.port = port;
    config.width = 64;
    config.height = 48;
    config.max_queued_bytes = 512U << 20;
    config.max_queue_age = std::chrono::hours(1);
    return config;
}

std::chrono::milliseconds time_destruction(std::unique_ptr<DirectSender> sender)
{
    const auto start = Clock::now();
    sender.reset();
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
}

} // namespace

TEST_CASE("destroying a sender whose connect never completes is bounded", "[direct_sender]")
{
    Listener listener(0);
    std::vector<int> clients;
    listener.saturate(clients);

    auto sender = std::make_unique<DirectSender>(config_for(listener.port()));
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // let the worker reach its connect

    CHECK(time_destruction(std::move(sender)) < kBound);
    for (const int fd : clients)
        ::close(fd);
}

TEST_CASE("destroying a sender blocked in send to a peer that does not read is bounded", "[direct_sender]")
{
    Listener listener(8);
    auto sender = std::make_unique<DirectSender>(config_for(listener.port()));
    REQUIRE(listener.accept_and_stall(2));

    // Far more than the socket buffers hold, so the worker ends up blocked in send().
    const std::vector<uint8_t> unit(1U << 20, 0x65);
    constexpr int kUnits = 100;
    for (int i = 0; i < kUnits; ++i)
    {
        EncodedVideoFrame frame;
        frame.data = unit.data();
        frame.size = unit.size();
        frame.keyframe = (i == 0);
        frame.frame_id = static_cast<uint32_t>(i);
        sender->submit(frame);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(sender->sent() < kUnits);

    CHECK(time_destruction(std::move(sender)) < kBound);
}

TEST_CASE("destroying a connected, idle sender is bounded", "[direct_sender]")
{
    Listener listener(8);
    auto sender = std::make_unique<DirectSender>(config_for(listener.port()));
    REQUIRE(listener.accept_and_stall(2));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    CHECK(time_destruction(std::move(sender)) < kBound);
}
