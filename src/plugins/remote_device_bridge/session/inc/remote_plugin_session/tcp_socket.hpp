// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transport.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace remote_plugin_session
{

struct Endpoint
{
    std::string host;
    uint16_t port = 0;
};

//! Parses "host[:port]"; a missing port falls back to default_port. Throws std::invalid_argument.
Endpoint parse_endpoint(const std::string& text, uint16_t default_port);

//! Minimal RAII wrapper over a blocking POSIX TCP socket (IPv4/IPv6 via getaddrinfo).
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
    explicit TcpSocket(int fd);
    ~TcpSocket();

    TcpSocket(TcpSocket&& other) noexcept;
    TcpSocket& operator=(TcpSocket&& other) noexcept;
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    //! Throws std::runtime_error when no address connects within the timeout.
    static TcpSocket connect(const Endpoint& endpoint, std::chrono::milliseconds timeout);
    //! Throws std::runtime_error when bind/listen fails. Port 0 picks an ephemeral port.
    static TcpSocket listen(const Endpoint& endpoint, int backlog);
    //! Returns an invalid socket when the timeout expires; throws on a hard error.
    TcpSocket accept(std::chrono::milliseconds timeout);

    bool valid() const;
    int fd() const;
    uint16_t local_port() const;
    //! Numeric "host:port" of the connected peer; "?" when unavailable.
    std::string peer_name() const;

    //! False when the peer closed or the send stalled past the socket send timeout.
    bool send_all(const uint8_t* data, size_t size);
    //! Timeout applies to the wait for the first byte; a stall mid-message reports Closed.
    RecvResult recv_all(uint8_t* data, size_t size, std::chrono::milliseconds timeout);

    //! Unblocks recv/send on other threads; the descriptor stays open until close().
    void shutdown();
    void close();

private:
    int m_fd = -1;
};

//! ITransport over one TCP stream; messages are delimited by the wire header's payload_size.
class TcpTransport final : public ITransport
{
public:
    explicit TcpTransport(TcpSocket socket);

    //! One transport per connection accepted on `listener`; null when the poll expires. `peer`, when given,
    //! receives the client's numeric host:port.
    static std::unique_ptr<ITransport> accept(TcpSocket& listener,
                                              std::chrono::milliseconds poll,
                                              std::string* peer = nullptr);

    bool send(const uint8_t* frame, size_t size) override;
    ReceiveResult receive(std::vector<uint8_t>& frame, std::chrono::milliseconds timeout) override;
    void shutdown() override;

private:
    TcpSocket m_socket;
};

} // namespace remote_plugin_session
