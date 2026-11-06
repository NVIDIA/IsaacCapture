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
    for (int* fd : { &m_connection, &m_control })
    {
        if (*fd >= 0)
            ::close(*fd);
        *fd = -1;
    }
    while (m_connection < 0 || m_control < 0)
    {
        const int fd = ::accept(m_listen, nullptr, nullptr);
        if (fd < 0)
            throw std::runtime_error(std::string("TcpReceiver: accept failed: ") + std::strerror(errno));
        const int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        uint8_t hello[4];
        if (!read_exact(fd, hello, sizeof(hello), true))
        {
            ::close(fd);
            continue;
        }
        if (std::memcmp(hello, "DATA", 4) == 0 && m_connection < 0)
            m_connection = fd;
        else if (std::memcmp(hello, "CTRL", 4) == 0 && m_control < 0)
            m_control = fd;
        else
            ::close(fd); // unknown or duplicate connection
    }
}

void TcpReceiver::stop_listening()
{
    if (m_listen >= 0)
        ::shutdown(m_listen, SHUT_RDWR); // makes a blocked accept() fail
}

void TcpReceiver::shutdown_session()
{
    for (const int fd : { m_connection, m_control })
    {
        if (fd >= 0)
            ::shutdown(fd, SHUT_RDWR);
    }
}

bool TcpReceiver::send_control(const uint8_t* data, size_t size)
{
    std::lock_guard<std::mutex> lock(m_control_send);
    size_t done = 0;
    while (done < size)
    {
        const ssize_t n = ::send(m_control, data + done, size - done, MSG_NOSIGNAL);
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
        return read_exact(m_control, out, size, true);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

size_t TcpReceiver::pending_bytes() const
{
    int pending = 0;
    if (m_connection < 0 || ::ioctl(m_connection, FIONREAD, &pending) < 0)
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
    uint8_t prefix[4];
    if (!read_exact(m_connection, prefix, 4, true))
        return false;
    const uint32_t length = avs1::read_length(prefix);
    if (length == 0 || length > avs1::kMaxRecordBytes)
        throw std::runtime_error("TcpReceiver: invalid record length " + std::to_string(length));
    record.resize(length);
    read_exact(m_connection, record.data(), length, false);
    return true;
}

} // namespace video_stage
