// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/tcp_socket.hpp"

#include "inc/remote_plugin_session/wire_codec.hpp"

#include <arpa/inet.h>
#include <log_bridge/logger.hpp>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <oxr_utils/os_time.hpp>
#include <sys/socket.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace remote_plugin_session
{

namespace
{

// A send may stall as long as the peer timeout tolerates silence; liveness is decided by the receive side.
constexpr std::chrono::milliseconds kSendTimeout{ 3000 };
constexpr std::chrono::milliseconds kMidMessageStall{ 3000 };
// Best effort: the kernel clamps to its rmem/wmem maximum; the effective sizes are logged once.
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

void configure_stream(int fd)
{
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval send_timeout{};
    send_timeout.tv_sec = static_cast<time_t>(kSendTimeout.count() / 1000);
    send_timeout.tv_usec = static_cast<suseconds_t>((kSendTimeout.count() % 1000) * 1000);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));

    int snd = kSocketBufferBytes;
    int rcv = kSocketBufferBytes;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &snd, sizeof(snd));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcv, sizeof(rcv));
    static std::atomic<bool> logged{ false };
    socklen_t len = sizeof(snd);
    if (!logged.exchange(true) && getsockopt(fd, SOL_SOCKET, SO_SNDBUF, &snd, &len) == 0 &&
        getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcv, &len) == 0)
    {
        isaaccapture::Logger::get("isaaccapture.plugins.remote_device_bridge.TcpSocket")
            ->info("[tcp] socket buffers: SO_SNDBUF={} SO_RCVBUF={} (requested {})", snd, rcv, kSocketBufferBytes);
    }
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

struct AddrInfoList
{
    addrinfo* head = nullptr;
    ~AddrInfoList()
    {
        if (head)
        {
            freeaddrinfo(head);
        }
    }
};

AddrInfoList resolve(const Endpoint& endpoint, bool passive)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = passive ? (AI_PASSIVE | AI_NUMERICSERV) : AI_NUMERICSERV;
    AddrInfoList list;
    const std::string service = std::to_string(endpoint.port);
    const int rc =
        getaddrinfo(endpoint.host.empty() ? nullptr : endpoint.host.c_str(), service.c_str(), &hints, &list.head);
    if (rc != 0)
    {
        throw std::runtime_error("getaddrinfo(" + endpoint.host + "): " + gai_strerror(rc));
    }
    return list;
}

//! Non-blocking connect with a deadline; on failure `last_error` says why and the caller tries the next address.
bool try_connect(TcpSocket& socket, const addrinfo& ai, std::chrono::milliseconds timeout, std::string& last_error)
{
    set_blocking(socket.fd(), false);
    const int rc = ::connect(socket.fd(), ai.ai_addr, ai.ai_addrlen);
    if (rc < 0 && errno != EINPROGRESS)
    {
        last_error = errno_string("connect");
        return false;
    }
    if (rc < 0)
    {
        const int ready = poll_fd(socket.fd(), POLLOUT, timeout);
        if (ready <= 0)
        {
            last_error = ready == 0 ? "connect timed out" : errno_string("poll");
            return false;
        }
        int so_error = 0;
        socklen_t len = sizeof(so_error);
        if (getsockopt(socket.fd(), SOL_SOCKET, SO_ERROR, &so_error, &len) < 0 || so_error != 0)
        {
            last_error = std::string("connect: ") + std::strerror(so_error != 0 ? so_error : errno);
            return false;
        }
    }
    set_blocking(socket.fd(), true);
    configure_stream(socket.fd());
    return true;
}

bool try_listen(TcpSocket& socket, const addrinfo& ai, int backlog, std::string& last_error)
{
    const int one = 1;
    setsockopt(socket.fd(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(socket.fd(), ai.ai_addr, ai.ai_addrlen) < 0)
    {
        last_error = errno_string("bind");
        return false;
    }
    if (::listen(socket.fd(), backlog) < 0)
    {
        last_error = errno_string("listen");
        return false;
    }
    return true;
}

//! Runs `attempt` on a fresh socket for each resolved address; the first success wins. `what` names the caller.
template <typename Attempt>
TcpSocket for_each_address(const Endpoint& endpoint, bool passive, const char* what, Attempt attempt)
{
    const AddrInfoList addresses = resolve(endpoint, passive);
    std::string last_error = "no addresses";

    for (const addrinfo* ai = addresses.head; ai != nullptr; ai = ai->ai_next)
    {
        TcpSocket socket(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
        if (!socket.valid())
        {
            last_error = errno_string("socket");
            continue;
        }
        if (attempt(socket, *ai, last_error))
        {
            return socket;
        }
    }

    throw std::runtime_error(std::string("TcpSocket::") + what + "(" + endpoint.host + ":" +
                             std::to_string(endpoint.port) + ") failed: " + last_error);
}

} // namespace

Endpoint parse_endpoint(const std::string& text, uint16_t default_port)
{
    Endpoint endpoint;
    endpoint.port = default_port;
    const size_t colon = text.rfind(':');
    if (colon == std::string::npos)
    {
        endpoint.host = text;
    }
    else
    {
        endpoint.host = text.substr(0, colon);
        const std::string port_text = text.substr(colon + 1);
        try
        {
            const unsigned long port = std::stoul(port_text);
            if (port_text.empty() || port == 0 || port > 65535)
            {
                throw std::out_of_range("port");
            }
            endpoint.port = static_cast<uint16_t>(port);
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument("invalid port in endpoint '" + text + "'");
        }
    }
    if (endpoint.host.empty())
    {
        throw std::invalid_argument("missing host in endpoint '" + text + "'");
    }
    return endpoint;
}

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
                            [timeout](TcpSocket& socket, const addrinfo& ai, std::string& last_error)
                            { return try_connect(socket, ai, timeout, last_error); });
}

TcpSocket TcpSocket::listen(const Endpoint& endpoint, int backlog)
{
    return for_each_address(endpoint, true, "listen",
                            [backlog](TcpSocket& socket, const addrinfo& ai, std::string& last_error)
                            { return try_listen(socket, ai, backlog, last_error); });
}

TcpSocket TcpSocket::accept(std::chrono::milliseconds timeout)
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
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNABORTED || errno == EINTR)
        {
            return TcpSocket();
        }
        throw std::runtime_error(errno_string("accept"));
    }
    TcpSocket client(fd);
    configure_stream(fd);
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
        if (n == 0)
        {
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

TcpSocket::RecvResult TcpSocket::recv_all(uint8_t* data, size_t size, std::chrono::milliseconds timeout)
{
    size_t received = 0;
    while (received < size)
    {
        const int ready = poll_fd(m_fd, POLLIN, received == 0 ? timeout : kMidMessageStall);
        if (ready == 0)
        {
            return received == 0 ? RecvResult::Timeout : RecvResult::Closed;
        }
        if (ready < 0)
        {
            return RecvResult::Closed;
        }
        const ssize_t n = ::recv(m_fd, data + received, size - received, 0);
        if (n < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
            {
                continue;
            }
            return RecvResult::Closed;
        }
        if (n == 0)
        {
            return RecvResult::Closed;
        }
        received += static_cast<size_t>(n);
    }
    return RecvResult::Ok;
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

TcpTransport::TcpTransport(TcpSocket socket) : m_socket(std::move(socket))
{
}

bool TcpTransport::send(const uint8_t* frame, size_t size)
{
    return m_socket.send_all(frame, size);
}

ReceiveResult TcpTransport::receive(std::vector<uint8_t>& frame, std::chrono::milliseconds timeout)
{
    frame.resize(wire::kHeaderSize);
    switch (m_socket.recv_all(frame.data(), wire::kHeaderSize, timeout))
    {
    case TcpSocket::RecvResult::Timeout:
        return ReceiveResult::Timeout;
    case TcpSocket::RecvResult::Closed:
        return ReceiveResult::Closed;
    case TcpSocket::RecvResult::Ok:
        break;
    }

    uint32_t payload_size = 0;
    try
    {
        payload_size = wire::decode_header(frame.data()).payload_size;
    }
    catch (const wire::DecodeError&)
    {
        return ReceiveResult::Malformed;
    }
    if (payload_size == 0)
    {
        return ReceiveResult::Ok;
    }
    frame.resize(wire::kHeaderSize + payload_size);
    // A header without its payload within the stall window is a broken peer, not a quiet one.
    return m_socket.recv_all(frame.data() + wire::kHeaderSize, payload_size, kMidMessageStall) ==
                   TcpSocket::RecvResult::Ok ?
               ReceiveResult::Ok :
               ReceiveResult::Closed;
}

void TcpTransport::shutdown()
{
    m_socket.shutdown();
}

std::unique_ptr<ITransport> TcpTransport::accept(TcpSocket& listener, std::chrono::milliseconds poll, std::string* peer)
{
    TcpSocket client = listener.accept(poll);
    if (!client.valid())
    {
        return nullptr;
    }
    if (peer != nullptr)
    {
        *peer = client.peer_name();
    }
    return std::make_unique<TcpTransport>(std::move(client));
}

} // namespace remote_plugin_session
