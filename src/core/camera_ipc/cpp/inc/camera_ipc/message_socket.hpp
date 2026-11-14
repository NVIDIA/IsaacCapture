// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unique_fd.hpp"

#include <chrono>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace core::camera
{

// A message carries at most this many descriptors.
inline constexpr std::size_t MAX_DESCRIPTORS_PER_MESSAGE = 64;

enum class SendStatus
{
    SENT,
    WOULD_BLOCK, // the peer's queue is full; nothing was sent
    CLOSED, // the peer is gone
};

enum class ReceiveStatus
{
    MESSAGE,
    TIMEOUT,
    CLOSED, // the peer is gone
};

struct ReceivedMessage
{
    ReceiveStatus status = ReceiveStatus::TIMEOUT;
    std::size_t bytes = 0; // of the payload, when status is MESSAGE
    std::vector<UniqueFd> descriptors; // the ones that came with the message
};

/**
 * @brief One end of a connection between two processes on this machine. It carries whole
 * messages in order, each with optional file descriptors.
 */
class MessageSocket
{
public:
    /**
     * @brief Connect to a MessageListener.
     * @throws std::system_error if nothing listens on the path.
     */
    static MessageSocket connect(const std::string& path);

    explicit MessageSocket(UniqueFd fd);

    /**
     * @brief Send one message. Never blocks.
     * @throws std::invalid_argument for an empty payload (it would read as a closed connection)
     * or more than MAX_DESCRIPTORS_PER_MESSAGE descriptors.
     * @throws std::system_error for a failure other than a full queue or a closed peer.
     */
    SendStatus send(std::span<const std::byte> payload, std::span<const int> descriptors = {});

    /**
     * @brief Wait up to `timeout` for one message and copy its payload into `buffer`.
     * @throws std::length_error if the message does not fit the buffer.
     * @throws std::system_error for a failure other than a closed peer.
     */
    ReceivedMessage receive(std::span<std::byte> buffer, std::chrono::milliseconds timeout);

    // For waiting on several sockets at once.
    int descriptor() const
    {
        return m_fd.get();
    }

private:
    UniqueFd m_fd;
};

/**
 * @brief Accepts connections on a socket file. Only processes of the same user can connect.
 *
 * A lock file next to the socket file (`<path>.lock`) decides which process owns the path, so
 * of two processes that start at the same moment exactly one listens. The lock file stays
 * behind, empty, for the next listener to lock.
 */
class MessageListener
{
public:
    /**
     * @brief Start listening on `path`. A socket file left behind by a process that has gone is
     * replaced; anything else found at the path is left alone.
     * @throws std::system_error with EADDRINUSE if another process listens on the path, with
     * EEXIST if the path holds something that is not a socket, or with the cause if the path
     * cannot be locked or bound.
     */
    explicit MessageListener(const std::string& path);

    // Removes the socket file.
    ~MessageListener();

    MessageListener(const MessageListener&) = delete;
    MessageListener& operator=(const MessageListener&) = delete;

    /**
     * @brief Wait up to `timeout` for a connection.
     * @return The connection, or nothing on timeout.
     */
    std::optional<MessageSocket> accept(std::chrono::milliseconds timeout);

    int descriptor() const
    {
        return m_fd.get();
    }

    const std::string& path() const
    {
        return m_path;
    }

private:
    UniqueFd m_lock; // declared first, so it is released after the socket is closed
    UniqueFd m_fd;
    std::string m_path;
};

} // namespace core::camera
