// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_link/tcp_transport.hpp"

#include "inc/remote_link/message_header.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>

#ifndef _WIN32
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <sys/types.h>

#    include <cerrno>
#    include <cstring>
#    include <fcntl.h>
#    include <netdb.h>
#    include <poll.h>
#    include <unistd.h>
#endif

namespace core
{

namespace
{

constexpr std::chrono::milliseconds kMidMessageStall{ 3000 };
// What a ws:// robot sends first, and the answer a WebSocket server gives a request it cannot upgrade.
constexpr std::string_view kHttpRequestStart = "GET ";
constexpr std::string_view kBadRequest = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

uint16_t parse_port(const std::string& port_text, const std::string& text)
{
    if (port_text.empty() || port_text.size() > 5 ||
        !std::all_of(port_text.begin(), port_text.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
    {
        throw std::invalid_argument("invalid port in endpoint '" + text + "'");
    }
    const unsigned long port = std::stoul(port_text);
    if (port > 65535)
    {
        throw std::invalid_argument("invalid port in endpoint '" + text + "'");
    }
    return static_cast<uint16_t>(port);
}

std::string host_port_text(const std::string& host, const std::string& port)
{
    return (host.find(':') != std::string::npos ? "[" + host + "]" : host) + ":" + port;
}

} // namespace

std::string endpoint_text(const Endpoint& endpoint)
{
    return host_port_text(endpoint.host, std::to_string(endpoint.port));
}

Endpoint parse_endpoint(const std::string& text, uint16_t default_port)
{
    Endpoint endpoint;
    endpoint.port = default_port;
    if (!text.empty() && text.front() == '[')
    {
        const size_t close = text.find(']');
        if (close == std::string::npos)
        {
            throw std::invalid_argument("missing ']' in endpoint '" + text + "'");
        }
        endpoint.host = text.substr(1, close - 1);
        const std::string rest = text.substr(close + 1);
        if (!rest.empty())
        {
            if (rest.front() != ':')
            {
                throw std::invalid_argument("unexpected text after ']' in endpoint '" + text + "'");
            }
            endpoint.port = parse_port(rest.substr(1), text);
        }
    }
    else if (std::count(text.begin(), text.end(), ':') > 1)
    {
        endpoint.host = text; // bare IPv6 literal
    }
    else
    {
        const size_t colon = text.find(':');
        endpoint.host = text.substr(0, colon);
        if (colon != std::string::npos)
        {
            endpoint.port = parse_port(text.substr(colon + 1), text);
        }
    }
    if (endpoint.host.empty())
    {
        throw std::invalid_argument("missing host in endpoint '" + text + "'");
    }
    return endpoint;
}

#ifndef _WIN32

namespace
{

// A send may stall as long as the peer timeout tolerates silence; liveness is decided by the receive side.
constexpr int kSendTimeoutMs = 3000;
constexpr int kSendBufferBytes = 256 * 1024;
// POLLOUT only while less than this is unsent, so latest-wins messages wait in the Link instead of the kernel.
constexpr int kNotSentLowWatermark = 64 * 1024;
constexpr int kListenBacklog = 16;

std::string errno_string(const char* what)
{
    return std::string(what) + ": " + std::strerror(errno);
}

void set_blocking(int fd, bool blocking)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK)) < 0)
    {
        throw std::runtime_error(errno_string("fcntl"));
    }
}

// Errors are ignored: on a non-TCP stream (socketpair) the TCP options do not apply.
void configure_stream(int fd)
{
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval send_timeout{};
    send_timeout.tv_sec = kSendTimeoutMs / 1000;
    send_timeout.tv_usec = (kSendTimeoutMs % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
    const int send_buffer = kSendBufferBytes;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
#    ifdef TCP_NOTSENT_LOWAT
    const int low_watermark = kNotSentLowWatermark;
    setsockopt(fd, IPPROTO_TCP, TCP_NOTSENT_LOWAT, &low_watermark, sizeof(low_watermark));
#    endif
}

//! Milliseconds for poll(): -1 for a negative (unlimited) timeout.
int poll_timeout(std::chrono::milliseconds timeout)
{
    return timeout.count() < 0 ? -1 : static_cast<int>(std::min<int64_t>(timeout.count(), INT_MAX));
}

int poll_fd(int fd, short events, std::chrono::milliseconds timeout)
{
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = events;
    int timeout_ms = poll_timeout(timeout);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{ std::max(timeout_ms, 0) };
    while (true)
    {
        const int result = ::poll(&pfd, 1, timeout_ms);
        if (result < 0 && errno == EINTR)
        {
            // A signal does not lengthen the wait.
            if (timeout_ms >= 0)
            {
                timeout_ms = poll_timeout(std::max(
                    std::chrono::milliseconds{ 0 },
                    std::chrono::ceil<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())));
            }
            continue;
        }
        if (result > 0 && (pfd.revents & (POLLERR | POLLNVAL)) != 0 && (pfd.revents & events) == 0)
        {
            return -1;
        }
        return result;
    }
}

struct AddrInfoList
{
    addrinfo* head = nullptr;
    AddrInfoList() = default;
    AddrInfoList(const AddrInfoList&) = delete;
    AddrInfoList& operator=(const AddrInfoList&) = delete;
    ~AddrInfoList()
    {
        if (head != nullptr)
        {
            freeaddrinfo(head);
        }
    }
};

void resolve(const Endpoint& endpoint, bool passive, AddrInfoList& list)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_NUMERICSERV | (passive ? AI_PASSIVE : 0);
    const std::string service = std::to_string(endpoint.port);
    const int rc = getaddrinfo(endpoint.host.c_str(), service.c_str(), &hints, &list.head);
    if (rc != 0)
    {
        throw std::runtime_error("cannot resolve '" + endpoint.host + "': " + gai_strerror(rc));
    }
}

std::string numeric_name(const sockaddr* addr, socklen_t len)
{
    char host[NI_MAXHOST];
    char service[NI_MAXSERV];
    if (getnameinfo(addr, len, host, sizeof(host), service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV) != 0)
    {
        return "?";
    }
    return host_port_text(host, service);
}

} // namespace

TcpSocket::TcpSocket(int fd) : m_fd(fd)
{
    if (fd < 0)
    {
        throw std::invalid_argument("TcpSocket: invalid descriptor");
    }
    configure_stream(fd);
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
    AddrInfoList addresses;
    resolve(endpoint, false, addresses);
    // The timeout bounds the whole call, not each address; a negative timeout waits without limit.
    const bool bounded = timeout.count() >= 0;
    const auto deadline = std::chrono::steady_clock::now() + (bounded ? timeout : std::chrono::milliseconds{ 0 });
    std::string last_error = "no addresses";
    bool timed_out = false; //!< whether last_error is a timeout
    for (const addrinfo* ai = addresses.head; ai != nullptr; ai = ai->ai_next)
    {
        if (bounded && ai != addresses.head && std::chrono::steady_clock::now() >= deadline)
        {
            break;
        }
        timed_out = false;
        const int fd = ::socket(ai->ai_family, ai->ai_socktype | SOCK_CLOEXEC, ai->ai_protocol);
        if (fd < 0)
        {
            last_error = errno_string("socket");
            continue;
        }
        TcpSocket socket;
        socket.m_fd = fd;
        set_blocking(fd, false);
        int rc = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc < 0 && errno != EINPROGRESS)
        {
            last_error = errno_string("connect");
            continue;
        }
        if (rc < 0)
        {
            std::chrono::milliseconds wait = timeout;
            if (bounded)
            {
                wait =
                    std::max(std::chrono::milliseconds{ 1 },
                             std::chrono::ceil<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()));
            }
            rc = poll_fd(fd, POLLOUT, wait);
            int so_error = 0;
            socklen_t len = sizeof(so_error);
            if (rc == 0)
            {
                last_error = "connect timed out";
                timed_out = true;
                continue;
            }
            if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) < 0 || so_error != 0)
            {
                last_error = std::string("connect: ") + std::strerror(so_error != 0 ? so_error : errno);
                continue;
            }
        }
        set_blocking(fd, true);
        configure_stream(fd);
        return socket;
    }
    // TeleopSession retries while connecting fails with "cannot connect to " (_remote.py).
    const std::string message = "cannot connect to " + endpoint_text(endpoint) + ": " + last_error;
    if (timed_out)
    {
        throw TcpConnectTimeoutError(message);
    }
    throw std::runtime_error(message);
}

bool TcpSocket::send_all(const uint8_t* data, size_t size)
{
    size_t sent = 0;
    while (sent < size)
    {
        const ssize_t n = ::send(m_fd, data + sent, size - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR)
        {
            continue;
        }
        if (n <= 0)
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
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
        {
            continue;
        }
        if (n <= 0)
        {
            return RecvResult::Closed;
        }
        received += static_cast<size_t>(n);
    }
    return RecvResult::Ok;
}

bool TcpSocket::wait_writable(std::chrono::milliseconds timeout)
{
    return m_fd >= 0 && poll_fd(m_fd, POLLOUT, timeout) > 0;
}

std::string TcpSocket::peer_name() const
{
    sockaddr_storage addr{};
    socklen_t len = sizeof(addr);
    if (getpeername(m_fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0)
    {
        return "?";
    }
    return numeric_name(reinterpret_cast<const sockaddr*>(&addr), len);
}

void TcpSocket::shutdown()
{
    if (m_fd >= 0)
    {
        ::shutdown(m_fd, SHUT_RDWR);
    }
}

void TcpSocket::shutdown_write()
{
    if (m_fd >= 0)
    {
        ::shutdown(m_fd, SHUT_WR);
    }
}

int TcpSocket::release()
{
    return std::exchange(m_fd, -1);
}

void TcpSocket::close()
{
    if (m_fd >= 0)
    {
        ::close(m_fd);
        m_fd = -1;
    }
}

Listener::Listener(const Endpoint& bind)
{
    AddrInfoList addresses;
    resolve(bind, true, addresses);
    std::string last_error = "no addresses";
    for (const addrinfo* ai = addresses.head; ai != nullptr; ai = ai->ai_next)
    {
        const int fd = ::socket(ai->ai_family, ai->ai_socktype | SOCK_CLOEXEC, ai->ai_protocol);
        if (fd < 0)
        {
            last_error = errno_string("socket");
            continue;
        }
        const int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (::bind(fd, ai->ai_addr, ai->ai_addrlen) < 0 || ::listen(fd, kListenBacklog) < 0)
        {
            last_error = errno_string("bind/listen");
            ::close(fd);
            continue;
        }
        m_fd = fd;
        return;
    }
    throw std::runtime_error("cannot listen on " + endpoint_text(bind) + ": " + last_error);
}

Listener::~Listener()
{
    close();
}

void Listener::close()
{
    if (m_fd >= 0)
    {
        ::close(std::exchange(m_fd, -1));
    }
}

uint16_t Listener::port() const
{
    if (m_fd < 0)
    {
        throw std::runtime_error("Listener is closed");
    }
    sockaddr_storage addr{};
    socklen_t len = sizeof(addr);
    if (getsockname(m_fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0)
    {
        throw std::runtime_error(errno_string("getsockname"));
    }
    if (addr.ss_family == AF_INET6)
    {
        return ntohs(reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_port);
    }
    return ntohs(reinterpret_cast<const sockaddr_in*>(&addr)->sin_port);
}

std::optional<Listener::Accepted> Listener::accept(std::chrono::milliseconds timeout)
{
    if (m_fd < 0)
    {
        throw std::runtime_error("Listener is closed");
    }
    const int ready = poll_fd(m_fd, POLLIN, timeout);
    if (ready == 0)
    {
        return std::nullopt;
    }
    if (ready < 0)
    {
        throw std::runtime_error(errno_string("poll(accept)"));
    }
    sockaddr_storage addr{};
    socklen_t len = sizeof(addr);
    const int fd = ::accept4(m_fd, reinterpret_cast<sockaddr*>(&addr), &len, SOCK_CLOEXEC);
    if (fd < 0)
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNABORTED || errno == EINTR)
        {
            return std::nullopt;
        }
        throw std::runtime_error(errno_string("accept"));
    }
    return Accepted{ fd, numeric_name(reinterpret_cast<const sockaddr*>(&addr), len) };
}

#else // _WIN32

namespace
{

[[noreturn]] void throw_unsupported()
{
    throw std::runtime_error("Remote sessions require POSIX sockets; not supported on Windows");
}

} // namespace

TcpSocket::TcpSocket(int /*fd*/)
{
    throw_unsupported();
}

TcpSocket::~TcpSocket() = default;

TcpSocket::TcpSocket(TcpSocket&& other) noexcept : m_fd(std::exchange(other.m_fd, -1))
{
}

TcpSocket& TcpSocket::operator=(TcpSocket&& other) noexcept
{
    m_fd = std::exchange(other.m_fd, -1);
    return *this;
}

TcpSocket TcpSocket::connect(const Endpoint& /*endpoint*/, std::chrono::milliseconds /*timeout*/)
{
    throw_unsupported();
}

bool TcpSocket::send_all(const uint8_t* /*data*/, size_t /*size*/)
{
    throw_unsupported();
}

TcpSocket::RecvResult TcpSocket::recv_all(uint8_t* /*data*/, size_t /*size*/, std::chrono::milliseconds /*timeout*/)
{
    throw_unsupported();
}

bool TcpSocket::wait_writable(std::chrono::milliseconds /*timeout*/)
{
    throw_unsupported();
}

std::string TcpSocket::peer_name() const
{
    throw_unsupported();
}

void TcpSocket::shutdown()
{
}

void TcpSocket::shutdown_write()
{
}

int TcpSocket::release()
{
    return std::exchange(m_fd, -1);
}

void TcpSocket::close()
{
}

Listener::Listener(const Endpoint& /*bind*/)
{
    throw_unsupported();
}

Listener::~Listener() = default;

void Listener::close()
{
}

uint16_t Listener::port() const
{
    throw_unsupported();
}

std::optional<Listener::Accepted> Listener::accept(std::chrono::milliseconds /*timeout*/)
{
    throw_unsupported();
}

#endif // _WIN32

int TcpSocket::fd() const
{
    return m_fd;
}

bool TcpSocket::valid() const
{
    return m_fd >= 0;
}

TcpTransport::TcpTransport(TcpSocket socket) : m_socket(std::move(socket))
{
}

bool TcpTransport::send(const uint8_t* message, size_t size)
{
    std::lock_guard<std::mutex> lock(m_send_mutex);
    return m_socket.send_all(message, size);
}

ReceiveResult TcpTransport::receive(std::vector<uint8_t>& message, std::chrono::milliseconds first_byte_timeout)
{
    message.resize(kHeaderSize);
    // The first byte on its own, so that a header cut off after it is Truncated, not Closed.
    switch (m_socket.recv_all(message.data(), 1, first_byte_timeout))
    {
    case TcpSocket::RecvResult::Timeout:
        return ReceiveResult::Timeout;
    case TcpSocket::RecvResult::Closed:
        return ReceiveResult::Closed;
    case TcpSocket::RecvResult::Ok:
        break;
    }
    if (m_socket.recv_all(message.data() + 1, kHeaderSize - 1, kMidMessageStall) != TcpSocket::RecvResult::Ok)
    {
        return ReceiveResult::Truncated;
    }
    uint32_t payload_size = 0;
    try
    {
        payload_size = decode_header(message.data()).payload_size;
    }
    catch (const LinkMessageDecodeError&)
    {
        if (!m_header_received && std::equal(kHttpRequestStart.begin(), kHttpRequestStart.end(), message.begin()))
        {
            std::lock_guard<std::mutex> lock(m_send_mutex);
            m_socket.send_all(reinterpret_cast<const uint8_t*>(kBadRequest.data()), kBadRequest.size());
            m_socket.shutdown();
        }
        return ReceiveResult::Malformed;
    }
    m_header_received = true;
    if (payload_size == 0)
    {
        return ReceiveResult::Ok;
    }
    message.resize(kHeaderSize + payload_size);
    // A header without its payload is a broken peer, not a quiet one: the stall limit applies from the first byte.
    return m_socket.recv_all(message.data() + kHeaderSize, payload_size, kMidMessageStall) == TcpSocket::RecvResult::Ok ?
               ReceiveResult::Ok :
               ReceiveResult::Truncated;
}

void TcpTransport::shutdown()
{
    m_socket.shutdown();
}

void TcpTransport::shutdown_write()
{
    std::lock_guard<std::mutex> lock(m_send_mutex);
    m_socket.shutdown_write();
}

bool TcpTransport::wait_writable(std::chrono::milliseconds timeout)
{
    return m_socket.wait_writable(timeout);
}

} // namespace core
