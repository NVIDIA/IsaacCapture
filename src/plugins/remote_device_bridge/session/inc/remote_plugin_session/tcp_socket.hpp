// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transport.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace remote_plugin_session
{

struct Endpoint
{
    std::string host;
    uint16_t port = 0;
};

//! Minimal RAII wrapper over a blocking POSIX TCP socket (IPv4/IPv6 via getaddrinfo).
class TcpSocket
{
public:
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
    //! Returns an invalid socket when the timeout expires; throws on a hard error. `peer`, when given, receives
    //! the client's numeric host:port.
    TcpSocket accept(std::chrono::milliseconds timeout, std::string* peer = nullptr);

    bool valid() const;
    int fd() const;
    uint16_t local_port() const;

    //! False when the peer closed or the send stalled past the socket send timeout.
    bool send_all(const uint8_t* data, size_t size);
    //! Timeout applies to the wait for the first byte; a stall mid-message reports Closed.
    //! Ok, Timeout or Closed.
    ReceiveResult recv_all(uint8_t* data, size_t size, std::chrono::milliseconds timeout);

    //! Unblocks recv/send on other threads; the descriptor stays open until close().
    void shutdown();
    void close();

private:
    //! Numeric "host:port" of the connected peer; "?" when unavailable.
    std::string peer_name() const;

    int m_fd = -1;
};

//! Accepts one connection on `listener` and wraps it in `Transport`; null when the poll expires. `peer`, when
//! given, receives the client's numeric host:port.
template <typename Transport>
std::unique_ptr<ITransport> accept_transport(TcpSocket& listener, std::chrono::milliseconds poll, std::string* peer)
{
    TcpSocket client = listener.accept(poll, peer);
    return client.valid() ? std::make_unique<Transport>(std::move(client)) : nullptr;
}

} // namespace remote_plugin_session
