// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tcp_socket.hpp"
#include "transport.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace remote_plugin_session
{

/*!
 * @brief RFC 6455 transport: each wire frame travels as one binary WebSocket message. No TLS, no extensions.
 *
 * The server role answers the HTTP upgrade inside its first receive(), so an accept loop never waits on a
 * slow client; send() fails until the upgrade is done. Incoming fragments are reassembled, pings answered.
 */
class WebSocketTransport final : public ITransport
{
public:
    //! Server role over an accepted socket; no I/O until the first receive() or wait_ready().
    explicit WebSocketTransport(TcpSocket socket);

    //! Client role: sends the upgrade for `path` with `host_header` ("host:port") and throws
    //! std::runtime_error unless the server answers 101.
    WebSocketTransport(TcpSocket socket,
                       const std::string& host_header,
                       const std::string& path,
                       std::chrono::milliseconds timeout = kConnectTimeout);

    //! Like TcpTransport::accept; the upgrade happens on the first receive().
    static std::unique_ptr<ITransport> accept(TcpSocket& listener,
                                              std::chrono::milliseconds poll,
                                              std::string* peer = nullptr);

    bool send(const uint8_t* frame, size_t size) override;
    ReceiveResult receive(std::vector<uint8_t>& frame, std::chrono::milliseconds timeout) override;
    void shutdown() override;
    //! Server: answers the HTTP upgrade if receive() has not yet.
    ReceiveResult wait_ready(std::chrono::milliseconds timeout) override;

private:
    enum class Role
    {
        Client,
        Server,
    };

    WebSocketTransport(TcpSocket socket, Role role);
    void client_upgrade(const std::string& host_header, const std::string& path, std::chrono::milliseconds timeout);
    bool send_message(uint8_t opcode, const uint8_t* data, size_t size);
    ReceiveResult accept_upgrade(std::chrono::milliseconds timeout);

    TcpSocket m_socket;
    Role m_role;
    std::atomic<bool> m_open{ false };
    std::mutex m_send_mutex; // receive() answers pings and closes concurrently with send()
    std::vector<uint8_t> m_send_buffer;
    std::mt19937 m_mask_rng;
};

} // namespace remote_plugin_session
