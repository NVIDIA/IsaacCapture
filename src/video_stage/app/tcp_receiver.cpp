// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tcp_receiver.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <video_stage/avs1.hpp>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace video_stage
{

namespace
{

// Reads exactly `size` bytes. Returns false only for end of stream before the first byte when `allow_eof`.
bool read_exact(int fd, uint8_t* out, size_t size, bool allow_eof)
{
    size_t done = 0;
    while (done < size)
    {
        const ssize_t n = ::recv(fd, out + done, size - done, 0);
        if (n == 0)
        {
            if (allow_eof && done == 0)
                return false;
            throw std::runtime_error("TcpReceiver: the connection ended inside a record");
        }
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            // A peer that exits with unread data resets the connection: at a record boundary that is the end of the
            // stream.
            if ((errno == ECONNRESET || errno == EPIPE) && allow_eof && done == 0)
                return false;
            throw std::runtime_error(std::string("TcpReceiver: recv failed: ") + std::strerror(errno));
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

} // namespace

TcpReceiver::TcpReceiver(uint16_t port)
{
    m_listen = ::socket(AF_INET, SOCK_STREAM, 0);
    if (m_listen < 0)
        throw std::runtime_error("TcpReceiver: cannot create a socket");
    const int one = 1;
    ::setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (::bind(m_listen, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 || ::listen(m_listen, 1) < 0)
    {
        const std::string why = std::strerror(errno);
        ::close(m_listen);
        throw std::runtime_error("TcpReceiver: cannot listen on port " + std::to_string(port) + ": " + why);
    }
}

int TcpReceiver::connection_fd() const
{
    std::lock_guard<std::mutex> lock(m_fd_mutex);
    return m_connection;
}

int TcpReceiver::control_fd() const
{
    std::lock_guard<std::mutex> lock(m_fd_mutex);
    return m_control;
}

TcpReceiver::~TcpReceiver()
{
    if (m_connection >= 0)
        ::close(m_connection);
    if (m_control >= 0)
        ::close(m_control);
    if (m_listen >= 0)
        ::close(m_listen);
}

void TcpReceiver::accept_session()
{
    {
        // Retire the earlier session under the lock a concurrent shutdown_session() takes, so it never sees a
        // descriptor that is being closed.
        std::lock_guard<std::mutex> lock(m_fd_mutex);
        for (int* fd : { &m_connection, &m_control })
        {
            if (*fd >= 0)
                ::close(*fd);
            *fd = -1;
        }
    }
    // Only this thread publishes descriptors, so it may read them here without the lock.
    while (m_connection < 0 || m_control < 0)
    {
        const int fd = ::accept(m_listen, nullptr, nullptr);
        if (fd < 0)
            throw std::runtime_error(std::string("TcpReceiver: accept failed: ") + std::strerror(errno));
        const int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        // Notice a link that died (Wi-Fi roam, unplugged cable) in seconds, as the sender does.
        ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
        const int idle = 5, interval = 2, count = 3;
        ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
        ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
        ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count));
        const unsigned user_timeout_ms = 5000;
        ::setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &user_timeout_ms, sizeof(user_timeout_ms));
        // A peer that connects and never finishes its hello must not hold the worker: bound the wait, and let a stop
        // reach this socket through m_pending.
        timeval timeout{ 2, 0 };
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        {
            std::lock_guard<std::mutex> lock(m_fd_mutex);
            m_pending = fd;
        }
        uint8_t hello[4];
        bool have_hello = false;
        try
        {
            have_hello = read_exact(fd, hello, sizeof(hello), true);
        }
        catch (const std::exception&)
        {
            have_hello = false;
        }
        timeval none{ 0, 0 };
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &none, sizeof(none));
        {
            // Hand the socket over (or close it) in the same critical section that stops advertising it as pending.
            std::lock_guard<std::mutex> lock(m_fd_mutex);
            m_pending = -1;
            if (have_hello && std::memcmp(hello, "DATA", 4) == 0 && m_connection < 0)
                m_connection = fd;
            else if (have_hello && std::memcmp(hello, "CTRL", 4) == 0 && m_control < 0)
                m_control = fd;
            else
                ::close(fd); // no hello, an unknown one, or a duplicate
        }
    }
}

void TcpReceiver::stop_listening()
{
    if (m_listen >= 0)
        ::shutdown(m_listen, SHUT_RDWR); // makes a blocked accept() fail
    std::lock_guard<std::mutex> lock(m_fd_mutex);
    if (m_pending >= 0)
        ::shutdown(m_pending, SHUT_RDWR); // and a hello being read
}

void TcpReceiver::shutdown_session()
{
    std::lock_guard<std::mutex> lock(m_fd_mutex);
    if (m_pending >= 0)
        ::shutdown(m_pending, SHUT_RDWR);
    for (const int fd : { m_connection, m_control })
    {
        if (fd >= 0)
            ::shutdown(fd, SHUT_RDWR);
    }
}

bool TcpReceiver::send_control(const uint8_t* data, size_t size)
{
    std::lock_guard<std::mutex> lock(m_control_send);
    const int fd = control_fd();
    size_t done = 0;
    while (done < size)
    {
        const ssize_t n = ::send(fd, data + done, size - done, MSG_NOSIGNAL);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool TcpReceiver::recv_control(uint8_t* out, size_t size)
{
    try
    {
        return read_exact(control_fd(), out, size, true);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

size_t TcpReceiver::pending_bytes() const
{
    int pending = 0;
    const int fd = connection_fd();
    if (fd < 0 || ::ioctl(fd, FIONREAD, &pending) < 0)
        return 0;
    return static_cast<size_t>(pending);
}

void TcpReceiver::request_keyframe()
{
    // Best effort: if the link is down the sender asks for a keyframe itself when it reconnects.
    const uint8_t request = 'K';
    (void)send_control(&request, 1);
}

bool TcpReceiver::read_record(std::vector<uint8_t>& record)
{
    const int fd = connection_fd();
    uint8_t prefix[4];
    if (!read_exact(fd, prefix, 4, true))
        return false;
    const uint32_t length = avs1::read_length(prefix);
    if (length == 0 || length > avs1::kMaxRecordBytes)
        throw std::runtime_error("TcpReceiver: invalid record length " + std::to_string(length));
    record.resize(length);
    read_exact(fd, record.data(), length, false);
    return true;
}

} // namespace video_stage
