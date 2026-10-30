// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/tcp_socket.hpp"

#include "inc/remote_plugin_session/transport.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <netdb.h>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace remote_plugin_session
{

namespace
{

// A send may stall as long as the peer timeout tolerates silence; liveness is decided by the receive side. It
// is independent of kConnectTimeout and kMidMessageStall, which only share its value.
constexpr std::chrono::milliseconds kSendTimeout{ 3000 };
// Best effort: the kernel clamps to its rmem/wmem maximum.
constexpr int kSocketBufferBytes = 4 * 1024 * 1024;

std::string errno_string(const char* what)
{
    return std::string(what) + ": " + std::strerror(errno);
}

void set_blocking(int fd, bool blocking)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0)
    {
        throw std::runtime_error(errno_string("fcntl(F_GETFL)"));
    }
    const int updated = blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK);
    if (fcntl(fd, F_SETFL, updated) < 0)
    {
        throw std::runtime_error(errno_string("fcntl(F_SETFL)"));
    }
}

template <typename T>
void set_option(int fd, int level, int name, const T& value, const char* what)
{
    if (setsockopt(fd, level, name, &value, sizeof(value)) < 0)
    {
        throw std::runtime_error(errno_string(what));
    }
}

void configure_stream(int fd)
{
    set_option(fd, IPPROTO_TCP, TCP_NODELAY, 1, "setsockopt(TCP_NODELAY)");
    timeval send_timeout{};
    send_timeout.tv_sec = static_cast<time_t>(kSendTimeout.count() / 1000);
    send_timeout.tv_usec = static_cast<suseconds_t>((kSendTimeout.count() % 1000) * 1000);
    set_option(fd, SOL_SOCKET, SO_SNDTIMEO, send_timeout, "setsockopt(SO_SNDTIMEO)");
    // Buffer sizes are a hint the kernel may clamp; a refusal is not an error.
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &kSocketBufferBytes, sizeof(kSocketBufferBytes));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &kSocketBufferBytes, sizeof(kSocketBufferBytes));
}

int poll_fd(int fd, short events, std::chrono::milliseconds timeout)
{
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = events;
    while (true)
    {
        const int result = ::poll(&pfd, 1, timeout.count() < 0 ? -1 : static_cast<int>(timeout.count()));
        if (result < 0 && errno == EINTR)
        {
            continue;
        }
        return result;
    }
}

using AddrInfoList = std::unique_ptr<addrinfo, decltype(&freeaddrinfo)>;

AddrInfoList resolve(const Endpoint& endpoint, bool passive)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = passive ? (AI_PASSIVE | AI_NUMERICSERV) : AI_NUMERICSERV;
    addrinfo* head = nullptr;
    const std::string service = std::to_string(endpoint.port);
    const int rc = getaddrinfo(endpoint.host.empty() ? nullptr : endpoint.host.c_str(), service.c_str(), &hints, &head);
    if (rc != 0)
    {
        throw std::runtime_error("getaddrinfo(" + endpoint.host + "): " + gai_strerror(rc));
    }
    return AddrInfoList(head, &freeaddrinfo);
}

//! Non-blocking connect with a deadline. Throws std::runtime_error with the reason.
void try_connect(TcpSocket& socket, const addrinfo& ai, std::chrono::milliseconds timeout)
{
    set_blocking(socket.fd(), false);
    const int rc = ::connect(socket.fd(), ai.ai_addr, ai.ai_addrlen);
    if (rc < 0 && errno != EINPROGRESS)
    {
        throw std::runtime_error(errno_string("connect"));
    }
    if (rc < 0)
    {
        const int ready = poll_fd(socket.fd(), POLLOUT, timeout);
        if (ready <= 0)
        {
            throw std::runtime_error(ready == 0 ? "connect timed out" : errno_string("poll"));
        }
        int so_error = 0;
        socklen_t len = sizeof(so_error);
        if (getsockopt(socket.fd(), SOL_SOCKET, SO_ERROR, &so_error, &len) < 0 || so_error != 0)
        {
            throw std::runtime_error(std::string("connect: ") + std::strerror(so_error != 0 ? so_error : errno));
        }
    }
    set_blocking(socket.fd(), true);
    configure_stream(socket.fd());
}

//! Throws std::runtime_error with the reason.
void try_listen(TcpSocket& socket, const addrinfo& ai, int backlog)
{
    set_option(socket.fd(), SOL_SOCKET, SO_REUSEADDR, 1, "setsockopt(SO_REUSEADDR)");
    if (::bind(socket.fd(), ai.ai_addr, ai.ai_addrlen) < 0)
    {
        throw std::runtime_error(errno_string("bind"));
    }
    if (::listen(socket.fd(), backlog) < 0)
    {
        throw std::runtime_error(errno_string("listen"));
    }
}

//! Runs `attempt` on a fresh socket for each resolved address; the first that does not throw wins. `what` names
//! the caller.
template <typename Attempt>
TcpSocket for_each_address(const Endpoint& endpoint, bool passive, const char* what, Attempt attempt)
{
    const AddrInfoList addresses = resolve(endpoint, passive);
    std::string errors;

    for (const addrinfo* ai = addresses.get(); ai != nullptr; ai = ai->ai_next)
    {
        try
        {
            const int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (fd < 0)
            {
                throw std::runtime_error(errno_string("socket"));
            }
            TcpSocket socket(fd);
            attempt(socket, *ai);
            return socket;
        }
        catch (const std::runtime_error& e)
        {
            errors += errors.empty() ? e.what() : std::string("; ") + e.what();
        }
    }

    throw std::runtime_error(std::string("TcpSocket::") + what + "(" + endpoint.host + ":" +
                             std::to_string(endpoint.port) + ") failed: " + (errors.empty() ? "no addresses" : errors));
}

} // namespace

TcpSocket::TcpSocket(int fd) : m_fd(fd)
{
}

TcpSocket::~TcpSocket()
{
    close();
}

TcpSocket::TcpSocket(TcpSocket&& other) noexcept : m_fd(std::exchange(other.m_fd, -1))
{
}

TcpSocket& TcpSocket::operator=(TcpSocket&& other) noexcept
{
    if (this != &other)
    {
        close();
        m_fd = std::exchange(other.m_fd, -1);
    }
    return *this;
}

TcpSocket TcpSocket::connect(const Endpoint& endpoint, std::chrono::milliseconds timeout)
{
    return for_each_address(endpoint, false, "connect",
                            [timeout](TcpSocket& socket, const addrinfo& ai) { try_connect(socket, ai, timeout); });
}

TcpSocket TcpSocket::listen(const Endpoint& endpoint, int backlog)
{
    return for_each_address(endpoint, true, "listen",
                            [backlog](TcpSocket& socket, const addrinfo& ai) { try_listen(socket, ai, backlog); });
}

TcpSocket TcpSocket::accept(std::chrono::milliseconds timeout, std::string* peer)
{
    const int ready = poll_fd(m_fd, POLLIN, timeout);
    if (ready == 0)
    {
        return TcpSocket();
    }
    if (ready < 0)
    {
        throw std::runtime_error(errno_string("poll(accept)"));
    }
    const int fd = ::accept(m_fd, nullptr, nullptr);
    if (fd < 0)
    {
        if (errno == ECONNABORTED || errno == EINTR)
        {
            return TcpSocket();
        }
        throw std::runtime_error(errno_string("accept"));
    }
    TcpSocket client(fd);
    configure_stream(fd);
    if (peer != nullptr)
    {
        *peer = client.peer_name();
    }
    return client;
}

bool TcpSocket::valid() const
{
    return m_fd >= 0;
}

int TcpSocket::fd() const
{
    return m_fd;
}

uint16_t TcpSocket::local_port() const
{
    sockaddr_storage addr{};
    socklen_t len = sizeof(addr);
    if (getsockname(m_fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0)
    {
        throw std::runtime_error(errno_string("getsockname"));
    }
    if (addr.ss_family == AF_INET)
    {
        return ntohs(reinterpret_cast<const sockaddr_in*>(&addr)->sin_port);
    }
    if (addr.ss_family == AF_INET6)
    {
        return ntohs(reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_port);
    }
    throw std::runtime_error("getsockname: unsupported address family");
}

std::string TcpSocket::peer_name() const
{
    sockaddr_storage addr{};
    socklen_t len = sizeof(addr);
    char host[NI_MAXHOST];
    char service[NI_MAXSERV];
    if (getpeername(m_fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0 ||
        getnameinfo(reinterpret_cast<const sockaddr*>(&addr), len, host, sizeof(host), service, sizeof(service),
                    NI_NUMERICHOST | NI_NUMERICSERV) != 0)
    {
        return "?";
    }
    return std::string(host) + ":" + service;
}

bool TcpSocket::send_all(const uint8_t* data, size_t size)
{
    size_t sent = 0;
    while (sent < size)
    {
        const ssize_t n = ::send(m_fd, data + sent, size - sent, MSG_NOSIGNAL);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

ReceiveResult TcpSocket::recv_all(uint8_t* data, size_t size, std::chrono::milliseconds timeout)
{
    size_t received = 0;
    while (received < size)
    {
        const int ready = poll_fd(m_fd, POLLIN, received == 0 ? timeout : kMidMessageStall);
        if (ready == 0)
        {
            return received == 0 ? ReceiveResult::Timeout : ReceiveResult::Closed;
        }
        if (ready < 0)
        {
            return ReceiveResult::Closed;
        }
        const ssize_t n = ::recv(m_fd, data + received, size - received, 0);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return ReceiveResult::Closed;
        }
        if (n == 0)
        {
            return ReceiveResult::Closed;
        }
        received += static_cast<size_t>(n);
    }
    return ReceiveResult::Ok;
}

void TcpSocket::shutdown()
{
    if (m_fd >= 0)
    {
        ::shutdown(m_fd, SHUT_RDWR);
    }
}

void TcpSocket::close()
{
    if (m_fd >= 0)
    {
        ::close(m_fd);
        m_fd = -1;
    }
}

} // namespace remote_plugin_session
