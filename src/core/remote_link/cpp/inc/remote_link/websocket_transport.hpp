// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tcp_transport.hpp"
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

namespace core
{

constexpr uint16_t kDefaultWebSocketPort = 47601;

struct RemoteAddress
{
    Endpoint endpoint;
    bool websocket = false;
    std::string path = "/";
};

//! "host:port" and "tcp://host:port" select TCP (default port 47600); "ws://host:port[/path]" selects WebSocket
//! (default port 47601). wss:// and other schemes throw std::invalid_argument.
RemoteAddress parse_remote(const std::string& text);

//! Connects and, for WebSocket, completes the upgrade within `timeout` (negative: no limit). Throws
//! std::runtime_error.
std::unique_ptr<ITransport> connect_transport(const RemoteAddress& remote, std::chrono::milliseconds timeout);

//! Wraps an accepted descriptor (ownership moves to the transport). A WebSocket upgrade happens in wait_ready()
//! or the first receive().
std::unique_ptr<ITransport> adopt_transport(int fd, bool websocket);

/*!
 * @brief RFC 6455 transport: one link message per binary WebSocket message. No TLS, no extensions.
 *
 * Clients mask, servers reject unmasked WebSocket frames (and clients masked ones). Pings are answered, a CLOSE is
 * echoed and then reported as Closed (Truncated between the fragments of a message), fragments are reassembled.
 */
class WebSocketTransport final : public ITransport
{
public:
    enum class Role
    {
        Client,
        Server,
    };

    //! Server: no I/O. Client: sends the upgrade for `host`/`path` and throws std::runtime_error unless the
    //! server answers 101 with the right Sec-WebSocket-Accept within `timeout`.
    WebSocketTransport(TcpSocket socket,
                       Role role,
                       const std::string& host = "",
                       const std::string& path = "/",
                       std::chrono::milliseconds timeout = std::chrono::milliseconds{ 3000 });

    //! Client: sends the upgrade request for `remote` without waiting for the answer; wait_upgraded() completes the
    //! upgrade. Throws PeerClosedError when the request cannot be sent.
    static std::unique_ptr<WebSocketTransport> start_client(TcpSocket socket, const RemoteAddress& remote);

    //! Client: reads the server's answer to the upgrade request. True once upgraded; false when no answer started
    //! within `timeout` (negative: no limit), and a later call may wait again. Throws PeerClosedError when the
    //! server closed or reset the connection before any byte of its answer, std::runtime_error when it refused the
    //! upgrade or answered badly.
    bool wait_upgraded(std::chrono::milliseconds timeout);

    //! False until the upgrade completed.
    bool send(const uint8_t* message, size_t size) override;
    ReceiveResult receive(std::vector<uint8_t>& message, std::chrono::milliseconds first_byte_timeout) override;
    void shutdown() override;
    //! Half-closes the TCP stream; no CLOSE frame is sent, the peer sees end of stream.
    void shutdown_write() override;
    //! Server: answers the HTTP upgrade if receive() has not yet. Client: wait_upgraded(), with a PeerClosedError
    //! reported as Closed and any other failed upgrade as Malformed.
    ReceiveResult wait_ready(std::chrono::milliseconds timeout) override;
    bool wait_writable(std::chrono::milliseconds timeout) override;

private:
    struct NoUpgradeYet
    {
    };
    WebSocketTransport(TcpSocket socket, Role role, NoUpgradeYet);

    void send_upgrade_request(const std::string& host, const std::string& path);
    bool send_frame(uint8_t opcode, const uint8_t* data, size_t size);
    ReceiveResult accept_upgrade(std::chrono::milliseconds timeout);

    TcpSocket m_socket;
    Role m_role;
    std::atomic<bool> m_open{ false };
    std::mutex m_send_mutex; // receive() answers PING and CLOSE concurrently with send()
    std::vector<uint8_t> m_send_buffer;
    std::mt19937 m_mask_rng;
    std::string m_upgrade_where; //!< client: "WebSocket upgrade to <host><path>", which starts every upgrade error
    std::string m_upgrade_accept; //!< client: the Sec-WebSocket-Accept the server must answer with
};

} // namespace core
