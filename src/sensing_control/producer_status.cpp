// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_control/producer_status.hpp"

#include <utility>

namespace plugins
{
namespace sensing
{

const char* to_string(ProducerState state)
{
    switch (state)
    {
    case ProducerState::Starting:
        return "starting";
    case ProducerState::Streaming:
        return "streaming";
    case ProducerState::CameraFault:
        return "camera_fault";
    case ProducerState::Error:
        return "error";
    case ProducerState::Stopping:
        return "stopping";
    }
    return "unknown";
}

ProducerStatus::ProducerStatus(std::string release) : m_release(std::move(release))
{
}

void ProducerStatus::set_state(ProducerState state, std::optional<Fault> fault)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state = state;
    // Only the failing states carry a fault; leaving one clears it.
    m_fault = (state == ProducerState::CameraFault || state == ProducerState::Error) ? std::move(fault) : std::nullopt;
}

void ProducerStatus::set_streams(std::vector<StreamStatus> streams)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_streams = std::move(streams);
}

bool ProducerStatus::set_direct_target(const std::string& track, const std::string& host, std::uint16_t port)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& stream : m_streams)
    {
        if (stream.track == track)
        {
            stream.direct_host = host;
            stream.direct_port = host.empty() ? 0 : port;
            stream.direct_connected = false;
            return true;
        }
    }
    return false;
}

StatusSnapshot ProducerStatus::snapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    StatusSnapshot out;
    out.state = m_state;
    out.fault = m_fault;
    out.streams = m_streams;
    out.release = m_release;
    out.uptime_seconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_started).count());
    return out;
}

} // namespace sensing
} // namespace plugins
