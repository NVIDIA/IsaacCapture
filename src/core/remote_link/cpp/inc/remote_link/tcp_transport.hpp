// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transport.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace core
{

constexpr uint16_t kDefaultTcpPort = 47600;

struct Endpoint
{
    std::string host;
    uint16_t port = 0;
};

//! Parses "host[:port]", "[v6]:port" or "[v6]"; a bare IPv6 literal takes the default port. Port 0 is allowed
//! (bind to an ephemeral port). Throws std::invalid_argument.
Endpoint parse_endpoint(const std::string& text, uint16_t default_port);

//! "host:port", with an IPv6 literal in brackets ("[::1]:47600"), as parse_endpoint() reads it back.
std::string endpoint_text(const Endpoint& endpoint);

//! TcpSocket::connect ran out of time before any address accepted the connection.
class TcpConnectTimeoutError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

//! Owning RAII wrapper over a connected stream socket. POSIX only; on Windows every operation throws.
class TcpSocket
{
public:
    enum class RecvResult
    {
        Ok,
        Timeout,
        Closed,
    };

    TcpSocket() = default;
    //! Adopts `fd` and applies the stream options (TCP_NODELAY, send timeout, buffer sizes).
    explicit TcpSocket(int fd);
    ~TcpSocket();

    TcpSocket(TcpSocket&& other) noexcept;
    TcpSocket& operator=(TcpSocket&& other) noexcept;
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    //! Non-blocking connect to each resolved address in turn; `timeout` bounds the connect, not the name lookup
    //! (negative: no limit). Throws TcpConnectTimeoutError when the last address tried timed out, else
    //! std::runtime_error.
    static TcpSocket connect(const Endpoint& endpoint, std::chrono::milliseconds timeout);

    //! False when the peer is gone or the send stalled past the send timeout.
    bool send_all(const uint8_t* data, size_t size);
    //! The timeout covers the first byte; a stall mid-read reports Closed.
    RecvResult recv_all(uint8_t* data, size_t size, std::chrono::milliseconds timeout);
    //! True when the socket polls writable (with TCP_NOTSENT_LOWAT: little unsent data queued).
    bool wait_writable(std::chrono::milliseconds timeout);
    //! Numeric "host:port" of the peer (an IPv6 host in brackets), or "?".
    std::string peer_name() const;
    //! Unblocks send/recv on other threads; the descriptor stays open until destruction.
    void shutdown();
    //! Sends end of stream after the bytes already written; receiving continues.
    void shutdown_write();
    //! Gives up ownership of the descriptor.
    int release();
    int fd() const;
    bool valid() const;

private:
    void close();

    int m_fd = -1;
};

//! ITransport over one stream; messages are delimited by the header's payload_size.
class TcpTransport final : public ITransport
{
public:
    explicit TcpTransport(TcpSocket socket);

    bool send(const uint8_t* message, size_t size) override;
    //! An HTTP request in place of the first message (a ws:// robot on the TCP port) is answered with HTTP 400,
    //! then the stream is shut down and Malformed returned, so that robot fails at once instead of retrying.
    ReceiveResult receive(std::vector<uint8_t>& message, std::chrono::milliseconds first_byte_timeout) override;
    void shutdown() override;
    void shutdown_write() override;
    bool wait_writable(std::chrono::milliseconds timeout) override;

private:
    TcpSocket m_socket;
    std::mutex m_send_mutex;
    bool m_header_received = false; //!< receive() thread only
};

//! Listening socket. POSIX only; on Windows the constructor throws.
class Listener
{
public:
    struct Accepted
    {
        int fd = -1; //!< owned by the caller
        std::string peer; //!< as TcpSocket::peer_name()
    };

    //! Binds and listens; port 0 picks an ephemeral port. Throws std::runtime_error.
    explicit Listener(const Endpoint& bind);
    ~Listener();
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;

    uint16_t port() const;
    //! nullopt when no connection arrives within `timeout`; throws on a hard error or once closed.
    std::optional<Accepted> accept(std::chrono::milliseconds timeout);
    //! Stops listening: connections not accepted yet are reset and new ones refused. Idempotent. Not while
    //! accept() runs on another thread.
    void close();

private:
    int m_fd = -1;
};

} // namespace core
