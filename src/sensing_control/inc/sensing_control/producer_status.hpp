// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

enum class ProducerState
{
    /// Opening the cameras and joining the room.
    Starting,
    /// Cameras are capturing and tracks are published.
    Streaming,
    /// The cameras could not be started; the producer retries and says why.
    CameraFault,
    /// Another failure (for example the LiveKit room was lost); the producer retries.
    Error,
    Stopping,
};

const char* to_string(ProducerState state);

/// A failure as an operator should read it: a stable code and a sentence that says what to check.
struct Fault
{
    std::string code;
    std::string message;
};

struct StreamStatus
{
    std::string track;
    std::uint32_t sensor_id = 0;
    /// Where this stream's encoded video is also sent (the PC video stage); empty host means no direct link.
    std::string direct_host;
    std::uint16_t direct_port = 0;
    bool direct_connected = false;
    /// Why the direct link is not up (for example "cannot connect to 192.168.1.5:5600: Connection refused").
    std::string direct_error;
    std::uint64_t frames = 0;
    std::uint64_t published = 0;
};

struct StatusSnapshot
{
    ProducerState state = ProducerState::Starting;
    std::optional<Fault> fault;
    std::vector<StreamStatus> streams;
    std::string release;
    std::uint64_t uptime_seconds = 0;
};

/// What the management channel reports. Written by the producer's main loop, read by the channel's thread.
class ProducerStatus
{
public:
    explicit ProducerStatus(std::string release);

    /// Set the state; a fault is kept only for the states that have one.
    void set_state(ProducerState state, std::optional<Fault> fault = std::nullopt);

    /// Replace the per-stream figures (the configured streams stay listed while the cameras are down).
    void set_streams(std::vector<StreamStatus> streams);

    /// Change one stream's direct-link target; false if no stream has that track name.
    bool set_direct_target(const std::string& track, const std::string& host, std::uint16_t port);

    StatusSnapshot snapshot() const;

private:
    mutable std::mutex m_mutex;
    std::string m_release;
    ProducerState m_state = ProducerState::Starting;
    std::optional<Fault> m_fault;
    std::vector<StreamStatus> m_streams;
    std::chrono::steady_clock::time_point m_started = std::chrono::steady_clock::now();
};

} // namespace sensing
} // namespace plugins
