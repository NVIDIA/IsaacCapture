/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ClockResponder.h"

#include "ClockExchange.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace argus_sender
{

struct ClockResponder::Impl
{
    int fd = -1;
    uint16_t port = 0;
    std::atomic<bool> stop{ false };
    std::atomic<uint64_t> answered{ 0 };
    std::thread thread;

    void run()
    {
        // One byte more than a request, so an oversized datagram is seen as oversized instead of truncated to look
        // valid.
        uint8_t buffer[kClockRequestBytes + 1];
        while (!stop.load())
        {
            sockaddr_in source;
            socklen_t sourceLength = sizeof(source);
            const ssize_t received =
                ::recvfrom(fd, buffer, sizeof(buffer), 0, reinterpret_cast<sockaddr*>(&source), &sourceLength);
            // As close to the arrival as the receive call allows.
            const int64_t t2 = monotonicClockNs();
            if (received < 0)
                continue; // the receive timeout, or a signal: look at the stop flag again
            uint32_t seq = 0;
            int64_t t1 = 0;
            if (!parseClockRequest(buffer, static_cast<size_t>(received), &seq, &t1))
                continue;
            const std::string reply = buildClockReply(seq, t1, t2, monotonicClockNs());
            if (::sendto(fd, reply.data(), reply.size(), 0, reinterpret_cast<const sockaddr*>(&source), sourceLength) ==
                static_cast<ssize_t>(reply.size()))
            {
                ++answered;
            }
        }
    }
};

ClockResponder::ClockResponder(const ClockResponderConfig& config) : m_impl(new Impl())
{
    m_impl->fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (m_impl->fd < 0)
        throw std::runtime_error(std::string("clock responder: cannot create a socket: ") + std::strerror(errno));
    sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(config.port);
    if (::inet_pton(AF_INET, config.bindHost.c_str(), &address.sin_addr) != 1)
    {
        ::close(m_impl->fd);
        throw std::runtime_error("clock responder: \"" + config.bindHost + "\" is not an IPv4 address");
    }
    if (::bind(m_impl->fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
    {
        const std::string why = std::strerror(errno);
        ::close(m_impl->fd);
        throw std::runtime_error("clock responder: cannot bind " + config.bindHost + ":" + std::to_string(config.port) +
                                 ": " + why);
    }
    socklen_t length = sizeof(address);
    ::getsockname(m_impl->fd, reinterpret_cast<sockaddr*>(&address), &length);
    m_impl->port = ntohs(address.sin_port);
    // The thread looks at the stop flag between datagrams; the timeout bounds how long that takes.
    timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
    ::setsockopt(m_impl->fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    m_impl->thread = std::thread(&Impl::run, m_impl.get());
}

ClockResponder::~ClockResponder()
{
    m_impl->stop = true;
    if (m_impl->thread.joinable())
        m_impl->thread.join();
    ::close(m_impl->fd);
}

uint16_t ClockResponder::port() const
{
    return m_impl->port;
}

uint64_t ClockResponder::answered() const
{
    return m_impl->answered.load();
}

} // namespace argus_sender
