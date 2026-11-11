// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sender_clock/sender_clock_client.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <video_types/video_types.hpp>

#include <cerrno>
#include <cstring>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>

namespace camera_bridge
{

namespace
{

constexpr std::size_t kRequestSize = 13;
constexpr std::size_t kReplySize = 29;

void put_le(std::uint8_t* out, std::uint64_t value, std::size_t bytes)
{
    for (std::size_t i = 0; i < bytes; ++i)
        out[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF);
}

std::uint64_t get_le(const std::uint8_t* in, std::size_t bytes)
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes; ++i)
        value |= static_cast<std::uint64_t>(in[i]) << (8 * i);
    return value;
}

} // namespace

SenderClockClient::SenderClockClient(SenderClockConfig config) : m_config(std::move(config))
{
}

SenderClockClient::~SenderClockClient()
{
    stop();
}

void SenderClockClient::start()
{
    if (m_running.exchange(true))
        return;
    m_thread = std::thread([this] { run(); });
}

void SenderClockClient::stop()
{
    m_running = false;
    if (m_thread.joinable())
        m_thread.join();
}

void SenderClockClient::reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_generation;
    m_estimator.reset();
}

void SenderClockClient::begin_session(uint64_t session_id)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_session_id && *m_session_id == session_id)
        return;
    // A change of session: every exchange so far was made before it. The first session also starts clean, so a
    // mapping measured before any frame arrived is not trusted for frames of a session that began later.
    m_session_id = session_id;
    ++m_generation;
    m_estimator.reset();
}

SenderClockSnapshot SenderClockClient::snapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    SenderClockSnapshot out;
    out.mapping = m_estimator.mapping(monotonic_ns());
    out.discontinuity = m_estimator.discontinuity();
    out.epoch = m_estimator.epoch();
    out.accepted_in_epoch = m_estimator.accepted_in_epoch();
    out.exchanges = m_exchanges;
    out.rejected = m_rejected;
    out.ignored = m_ignored;
    out.timeouts = m_timeouts;
    out.generation = m_generation;
    out.last_error = m_last_error;
    return out;
}

void SenderClockClient::run()
{
    sockaddr_in peer{};
    peer.sin_family = AF_INET;
    peer.sin_port = htons(m_config.port);
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0 || inet_pton(AF_INET, m_config.host.c_str(), &peer.sin_addr) != 1)
    {
        if (fd >= 0)
            ::close(fd);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_last_error = "sender clock host must be an IPv4 address: " + m_config.host;
        m_running = false;
        return;
    }

    std::uint32_t sequence = 0;
    while (m_running)
    {
        ++sequence;
        std::uint8_t request[kRequestSize];
        request[0] = 'Q';
        put_le(request + 1, sequence, 4);
        uint64_t generation;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            generation = m_generation;
        }
        // t1 is read as late as possible, immediately before the send.
        const int64_t t1 = monotonic_ns();
        put_le(request + 5, static_cast<std::uint64_t>(t1), 8);
        const ssize_t sent = ::sendto(fd, request, sizeof(request), 0, reinterpret_cast<sockaddr*>(&peer), sizeof(peer));

        bool answered = false;
        if (sent == static_cast<ssize_t>(sizeof(request)))
        {
            const auto deadline = std::chrono::steady_clock::now() + m_config.reply_timeout;
            while (!answered && m_running)
            {
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
                if (remaining.count() <= 0)
                    break;
                pollfd wait{ fd, POLLIN, 0 };
                const int ready = ::poll(&wait, 1, static_cast<int>(std::min<int64_t>(remaining.count(), 50)));
                if (ready <= 0)
                    continue;
                std::uint8_t reply[64];
                sockaddr_in from{};
                socklen_t from_len = sizeof(from);
                const ssize_t got =
                    ::recvfrom(fd, reply, sizeof(reply), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
                const int64_t t4 = monotonic_ns(); // first, so it is as close to the receipt as software allows
                if (got < 0)
                    continue;
                std::lock_guard<std::mutex> lock(m_mutex);
                // A datagram that is not the answer to this request is not a measurement; neither is one sent
                // before the last reset.
                if (from.sin_addr.s_addr != peer.sin_addr.s_addr || from.sin_port != peer.sin_port ||
                    got != static_cast<ssize_t>(kReplySize) || reply[0] != 'R' || get_le(reply + 1, 4) != sequence ||
                    static_cast<int64_t>(get_le(reply + 5, 8)) != t1 || generation != m_generation)
                {
                    ++m_ignored;
                    continue;
                }
                Exchange exchange;
                exchange.t1 = t1;
                exchange.t2 = static_cast<int64_t>(get_le(reply + 13, 8));
                exchange.t3 = static_cast<int64_t>(get_le(reply + 21, 8));
                exchange.t4 = t4;
                ++m_exchanges;
                if (m_estimator.add(exchange))
                    ++m_rejected;
                m_last_error.clear();
                answered = true;
            }
        }
        if (!answered)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_timeouts;
            m_last_error = sent == static_cast<ssize_t>(sizeof(request)) ? "no clock reply" : std::strerror(errno);
        }
        // Sleep in small steps so a stop request is honoured promptly.
        const auto wake = std::chrono::steady_clock::now() + m_config.interval;
        while (m_running && std::chrono::steady_clock::now() < wake)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ::close(fd);
}

} // namespace camera_bridge
