// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <camera_core/publisher.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace core::camera
{

/**
 * @brief Serves a Publisher's groups to readers in other processes.
 *
 * Listens on a socket file. A reader's first message says what it came for, and the answer is
 * the first thing it receives: what is published, or the group it asked for with the
 * descriptors of the group's pools. After that an in-order reader is sent each frame set as
 * it is published and a newest-order reader one each time it asks. A reader that goes away,
 * cleanly or not, has every frame set it held taken back.
 */
class Endpoint
{
public:
    /**
     * @brief Start listening on `path`. Readers are answered once start() has been called.
     * @param publisher Must outlive the endpoint.
     * @throws std::system_error if another process listens on the path or it cannot be bound.
     */
    Endpoint(Publisher& publisher, const std::string& path);

    // Stops serving.
    ~Endpoint();

    Endpoint(const Endpoint&) = delete;
    Endpoint& operator=(const Endpoint&) = delete;

    /**
     * @brief Name what a reader maps for a group: one descriptor per stream, in stream order,
     * for the stream's pool. The endpoint does not own them; they stay open while it serves.
     * A group without them cannot be subscribed to.
     * @throws std::logic_error after start().
     * @throws std::out_of_range for a group the publisher does not have.
     * @throws std::invalid_argument if the count does not match the group's streams.
     */
    void set_group_memory(GroupId group, std::vector<int> descriptors);

    /**
     * @brief Serve readers. After a stop() this takes the path again.
     * @throws std::system_error if another process took the path meanwhile.
     */
    void start();

    /**
     * @brief Stop serving: every subscription ends, every reader sees its connection close, and
     * the path is given up, so a reader that tries to connect is refused at once. Does nothing
     * if the endpoint is not serving.
     */
    void stop();

    // Readers connected right now.
    std::size_t reader_count() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace core::camera
