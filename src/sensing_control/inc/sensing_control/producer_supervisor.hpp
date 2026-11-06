// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "control_handler.hpp"
#include "producer_status.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plugins
{
namespace sensing
{

/// Where a stream's encoded video goes in addition to LiveKit; chosen at runtime over the management channel.
struct DirectTarget
{
    std::string host;
    std::uint16_t port = 0;
};

using DirectTargets = std::map<std::string, DirectTarget>; ///< by track name

/// A start-up failure the factory has already explained: the state to report and the fault to show.
class StartupFailure : public std::runtime_error
{
public:
    StartupFailure(Fault fault, ProducerState state)
        : std::runtime_error(fault.message), m_fault(std::move(fault)), m_state(state)
    {
    }
    const Fault& fault() const
    {
        return m_fault;
    }
    ProducerState state() const
    {
        return m_state;
    }

private:
    Fault m_fault;
    ProducerState m_state;
};

/// One running capture-and-publish pipeline. The real producer and the mock producer each provide one.
class CameraSession
{
public:
    virtual ~CameraSession() = default;

    /// Poll for frames and hand them on. Throws once the session can no longer work (room lost, camera gone).
    virtual void update() = 0;
    virtual void flush() = 0;
    /// Current per-stream figures, including each direct link's target and whether it is connected.
    virtual std::vector<StreamStatus> streams() const = 0;
    virtual void set_direct_target(const std::string& track, const std::string& host, std::uint16_t port) = 0;
};

/// Builds a session, applying `targets` over the configured direct links. Throws with the failure's text.
using SessionFactory = std::function<std::unique_ptr<CameraSession>(const DirectTargets& targets)>;

struct SupervisorConfig
{
    /// How long to wait before trying to start the cameras again after a failure.
    std::chrono::milliseconds retry_interval{ 5000 };
    std::chrono::milliseconds poll_interval{ 15 };
    /// How often the figures in the status are refreshed from the running session.
    std::chrono::milliseconds status_interval{ 1000 };
};

/**
 * @brief Keeps the producer running and reporting.
 *
 * A camera that cannot be started (no power, no link) is a state to report and retry, not a reason to exit: the
 * management channel stays up, says why in plain words, and the stream comes back when the cable is fixed. Direct
 * targets set over the channel are remembered here, so a restarted session keeps them. Implements the channel's
 * actions; they are safe to call from the channel's thread.
 */
class ProducerSupervisor : public ControlActions
{
public:
    /// `publish` receives each pushed notification as JSON text (the channel's broadcast); it may be empty.
    ProducerSupervisor(ProducerStatus& status,
                       std::vector<StreamStatus> configured,
                       SessionFactory factory,
                       std::function<void(const std::string&)> publish,
                       const SupervisorConfig& config = {});

    /// Run until `stop()` returns true.
    void run(const std::function<bool()>& stop);

    ActionResult set_direct_target(const std::string& track, const std::string& host, std::uint16_t port) override;

private:
    void sleep_unless_stopped(std::chrono::milliseconds duration, const std::function<bool()>& stop) const;
    void announce(ProducerState state, std::optional<Fault> fault);

    ProducerStatus& m_status;
    std::vector<StreamStatus> m_configured;
    SessionFactory m_factory;
    std::function<void(const std::string&)> m_publish;
    SupervisorConfig m_config;

    std::mutex m_mutex; ///< guards m_session and m_targets
    std::unique_ptr<CameraSession> m_session;
    DirectTargets m_targets;
};

} // namespace sensing
} // namespace plugins
