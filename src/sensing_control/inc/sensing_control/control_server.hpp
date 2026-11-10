// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace plugins
{
namespace sensing
{

struct ControlServerConfig
{
    std::string bind_address = "0.0.0.0";
    /// 0 picks a free port (tests); port() reports it.
    std::uint16_t port = 5610;
    /// Shared secret every client must present (`Authorization: Bearer` or `?token=`). Required: a channel that can
    /// redirect camera video is never offered without one.
    std::string token;
    std::size_t max_clients = 8;
    std::size_t max_message_bytes = 64 * 1024;
};

/// The first line of a secret file with surrounding whitespace removed; throws naming the file if it is unreadable or
/// empty. The secret is read from a file so it never appears in a process listing.
std::string read_secret_file(const std::string& path);

/**
 * @brief An authenticated WebSocket server on its own thread.
 *
 * Each received text message goes to `on_message` and its return value is sent back to that client; the callback
 * runs on the server thread and must not block. broadcast() pushes a text message to every connected client from
 * any thread. A client that sends a bad handshake, a wrong token, an oversized or malformed message, or that stops
 * reading is disconnected with a reason where the protocol allows one.
 */
class ControlServer
{
public:
    using MessageHandler = std::function<std::string(const std::string&)>;

    /// Starts listening. Throws std::runtime_error with the reason if the token is empty or the port cannot be bound.
    ControlServer(const ControlServerConfig& config, MessageHandler on_message);
    ~ControlServer();

    ControlServer(const ControlServer&) = delete;
    ControlServer& operator=(const ControlServer&) = delete;

    std::uint16_t port() const
    {
        return m_port;
    }

    /// Queue a text message for every connected client.
    void broadcast(const std::string& text);

    /// Clients that have completed the handshake.
    std::size_t client_count() const
    {
        return m_clients.load();
    }

private:
    struct Client;
    void run();
    void wake();

    ControlServerConfig m_config;
    MessageHandler m_on_message;
    int m_listen_fd = -1;
    int m_wake_fd = -1;
    std::uint16_t m_port = 0;
    std::atomic<bool> m_stop{ false };
    std::atomic<std::size_t> m_clients{ 0 };
    std::mutex m_mutex;
    std::vector<std::string> m_broadcasts; ///< guarded by m_mutex
    std::thread m_thread;
};

} // namespace sensing
} // namespace plugins
