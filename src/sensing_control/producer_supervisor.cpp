// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_control/producer_supervisor.hpp"

#include "inc/sensing_control/camera_fault.hpp"

#include <log_bridge/logger.hpp>

#include <thread>
#include <utility>

namespace plugins
{
namespace sensing
{

namespace
{

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.sensing.ProducerSupervisor");
}

} // namespace

ProducerSupervisor::ProducerSupervisor(ProducerStatus& status,
                                       std::vector<StreamStatus> configured,
                                       SessionFactory factory,
                                       std::function<void(const std::string&)> publish,
                                       const SupervisorConfig& config)
    : m_status(status),
      m_configured(std::move(configured)),
      m_factory(std::move(factory)),
      m_publish(std::move(publish)),
      m_config(config)
{
    // The configured direct targets are the starting point; a runtime change replaces them.
    for (const auto& stream : m_configured)
        m_targets[stream.track] = { stream.direct_host, stream.direct_port };
    m_status.set_streams(m_configured);
}

void ProducerSupervisor::announce(ProducerState state, std::optional<Fault> fault)
{
    m_status.set_state(state, fault);
    if (fault)
        logger()->error("{}: {}", fault->code, fault->message);
    if (m_publish)
    {
        const auto snapshot = m_status.snapshot();
        if (fault)
            m_publish(error_event(*fault));
        m_publish(state_event(snapshot));
    }
}

std::string ProducerSupervisor::watch(const std::vector<StreamStatus>& streams, std::chrono::steady_clock::time_point now)
{
    std::string stalled;
    for (const auto& stream : streams)
    {
        auto& entry = m_watch[stream.track];
        if (stream.frames != entry.frames)
        {
            entry.frames = stream.frames;
            entry.changed = now;
        }
        else if (m_config.stall_timeout.count() > 0 && now - entry.changed >= m_config.stall_timeout)
        {
            const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now - entry.changed).count();
            stalled += (stalled.empty() ? "" : "; ") + std::string("no frame for ") + std::to_string(seconds) +
                       " s from sensor " + std::to_string(stream.sensor_id) + " (stream " + stream.track + ")";
        }

        // A direct link that cannot reach the PC is reported once per distinct reason, not every second.
        if (stream.direct_error != entry.direct_error)
        {
            entry.direct_error = stream.direct_error;
            if (!stream.direct_error.empty() && m_publish)
            {
                m_publish(error_event(
                    { "direct_link_failed", "Stream " + stream.track + ": " + stream.direct_error +
                                                ". Check that the PC worker is running and listening on that port, "
                                                "and that the address is the PC's (set it with set_direct_target)." }));
            }
        }
    }
    return stalled;
}

void ProducerSupervisor::sleep_unless_stopped(std::chrono::milliseconds duration, const std::function<bool()>& stop) const
{
    const auto until = std::chrono::steady_clock::now() + duration;
    while (!stop() && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

ActionResult ProducerSupervisor::set_direct_target(const std::string& track, const std::string& host, std::uint16_t port)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_status.set_direct_target(track, host, port))
        return { false, "unknown_stream", "no stream named \"" + track + "\"" };
    m_targets[track] = { host, host.empty() ? std::uint16_t(0) : port };
    // A live session retargets now; otherwise the target is used when the cameras come up.
    if (m_session)
        m_session->set_direct_target(track, host, port);
    logger()->info(
        "stream {} now sends to {}", track, host.empty() ? std::string("nowhere") : host + ":" + std::to_string(port));
    if (m_publish)
        m_publish(state_event(m_status.snapshot()));
    return {};
}

void ProducerSupervisor::run(const std::function<bool()>& stop)
{
    // A fault that persists across retries is announced once, not on every attempt.
    std::optional<Fault> last_fault;
    bool first = true;
    while (!stop())
    {
        if (first || !last_fault)
            announce(ProducerState::Starting, std::nullopt);
        first = false;

        std::unique_ptr<CameraSession> session;
        DirectTargets targets;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            targets = m_targets;
        }
        try
        {
            session = m_factory(targets);
        }
        catch (const std::exception& e)
        {
            // The factory explains the failures it understands; anything else is read as a camera failure.
            const auto* explained = dynamic_cast<const StartupFailure*>(&e);
            const Diagnosis diagnosis =
                explained ? Diagnosis{ explained->fault(), explained->state() } : diagnose_failure(e.what());
            Fault fault = diagnosis.fault;
            const ProducerState state = diagnosis.state;
            if (!last_fault || last_fault->message != fault.message)
                announce(state, fault);
            last_fault = std::move(fault);
            sleep_unless_stopped(m_config.retry_interval, stop);
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_session = std::move(session);
        }
        last_fault.reset();
        announce(ProducerState::Streaming, std::nullopt);

        std::optional<Fault> failure;
        ProducerState failure_state = ProducerState::Error;
        m_watch.clear();
        for (const auto& stream : m_configured)
        {
            m_watch[stream.track].changed = std::chrono::steady_clock::now();
        }
        auto next_status = std::chrono::steady_clock::now();
        while (!stop() && !failure)
        {
            try
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_session->update();
                if (std::chrono::steady_clock::now() >= next_status)
                {
                    next_status += m_config.status_interval;
                    auto streams = m_session->streams();
                    // The session reports the live figures; the targets are the ones the operator last chose.
                    for (auto& stream : streams)
                    {
                        const auto found = m_targets.find(stream.track);
                        if (found != m_targets.end())
                        {
                            stream.direct_host = found->second.host;
                            stream.direct_port = found->second.port;
                        }
                    }
                    const std::string stalled = watch(streams, std::chrono::steady_clock::now());
                    m_status.set_streams(std::move(streams));
                    if (!stalled.empty())
                        throw std::runtime_error(stalled);
                }
            }
            catch (const std::exception& e)
            {
                failure_state = ProducerState::Error;
                Diagnosis diagnosis = diagnose_failure(e.what());
                if (diagnosis.fault.code == "startup_failed")
                {
                    // A session that was running and failed for an unrecognised reason did not "fail to start".
                    diagnosis = { { "stream_failed", std::string("The stream stopped. The producer restarts it "
                                                                 "automatically. (") +
                                                         e.what() + ")" },
                                  ProducerState::Error };
                }
                failure = diagnosis.fault;
                failure_state = diagnosis.state;
            }
            if (!failure)
                std::this_thread::sleep_for(m_config.poll_interval);
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_session)
            {
                try
                {
                    m_session->flush();
                }
                catch (const std::exception& e)
                {
                    logger()->warn("flush on shutdown failed: {}", e.what());
                }
                m_session.reset();
            }
        }
        if (failure)
        {
            announce(failure_state, failure);
            last_fault.reset();
            sleep_unless_stopped(m_config.retry_interval, stop);
        }
    }
    announce(ProducerState::Stopping, std::nullopt);
}

} // namespace sensing
} // namespace plugins
