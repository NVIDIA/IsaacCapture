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
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plugins
{
namespace sensing
{

/// Where a stream's encoded video goes (the PC video stage); chosen at startup or at runtime over the management
/// channel.
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
    /// A running session whose stream delivers no new frame for this long is failed and restarted. Zero disables it.
    std::chrono::milliseconds stall_timeout{ 5000 };
};

/**
 * @brief Keeps the producer running and reporting.
 *
 * A camera that cannot be started (no power, no link) is a state to report and retry, not a reason to exit: the
 * management channel stays up, says why in plain words, and the stream comes back when the cable is fixed.
 *
 * One thread, the one in run(), owns the session: it creates, updates, retargets, flushes and destroys it. The
 * channel's thread only records what was asked (set_direct_target) and run() applies it between updates, so a slow
 * camera start never blocks a command and a command never races a session that is being built or torn down. Direct
 * targets are remembered, so a restarted session starts with the latest.
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
    /// Compare the session's figures with the last ones: report new direct-link errors, and return a description of a
    /// stalled stream (empty when every stream is delivering).
    std::string watch(const std::vector<StreamStatus>& streams, std::chrono::steady_clock::time_point now);

    ProducerStatus& m_status;
    std::vector<StreamStatus> m_configured;
    SessionFactory m_factory;
    std::function<void(const std::string&)> m_publish;
    SupervisorConfig m_config;

    /// Per stream: the last frame count seen, when it last changed, and the direct-link error last reported.
    struct Watch
    {
        std::uint64_t frames = 0;
        std::chrono::steady_clock::time_point changed;
        std::string direct_error;
    };
    std::map<std::string, Watch> m_watch; ///< main thread only

    /// Applies the targets that changed since the session was built, on the owning thread.
    void apply_pending(CameraSession& session);

    std::mutex m_targets_mutex; ///< guards m_targets and m_pending
    DirectTargets m_targets; ///< what the operator last chose, by track
    std::set<std::string> m_pending; ///< tracks whose target changed since the running session last heard of it
};

} // namespace sensing
} // namespace plugins
