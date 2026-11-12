// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/chrony_clock/chrony_capture_clock.hpp"

#include <utility>

namespace camera_bridge
{

namespace
{

isaaccapture::timesync::ChronyMonitorConfig monitor_config(const ChronyCaptureClockConfig& c,
                                                           const std::string& host,
                                                           int port)
{
    isaaccapture::timesync::ChronyMonitorConfig m;
    m.chronyc_path = c.chronyc_path;
    m.host = host;
    m.port = port;
    m.poll_interval = c.poll_interval;
    m.command_timeout = c.command_timeout;
    return m;
}

isaaccapture::timesync::ClockReader resolve(isaaccapture::timesync::ClockReader clocks)
{
    if (!clocks.monotonic_ns || !clocks.realtime_ns)
        return isaaccapture::timesync::system_clock_reader();
    return clocks;
}

} // namespace

ChronyCaptureClock::ChronyCaptureClock(ChronyCaptureClockConfig config,
                                       isaaccapture::timesync::CommandRunner runner,
                                       isaaccapture::timesync::ClockReader clocks)
    : m_config(std::move(config)),
      m_clocks(resolve(std::move(clocks))),
      m_pc_monitor(monitor_config(m_config, m_config.pc_port != 0 ? "127.0.0.1" : "", m_config.pc_port),
                   runner,
                   m_clocks.monotonic_ns,
                   m_clocks.realtime_ns),
      m_sender_monitor(monitor_config(m_config, m_config.sender_host, m_config.sender_port),
                       runner,
                       m_clocks.monotonic_ns,
                       m_clocks.realtime_ns)
{
    m_qualifier_config.budget_ns = m_config.budget_ns;
    m_qualifier_config.expected_sender_reference_id = m_config.expected_sender_reference_id;
}

ChronyCaptureClock::~ChronyCaptureClock()
{
    stop();
}

void ChronyCaptureClock::start()
{
    m_pc_monitor.start();
    m_sender_monitor.start();
}

void ChronyCaptureClock::stop()
{
    m_pc_monitor.stop();
    m_sender_monitor.stop();
}

void ChronyCaptureClock::poll_once()
{
    m_pc_monitor.poll_once();
    m_sender_monitor.poll_once();
}

isaaccapture::timesync::Qualification ChronyCaptureClock::qualify(bool sender_valid,
                                                                  int64_t sender_realtime_ns,
                                                                  int64_t upstream_uncertainty_ns)
{
    isaaccapture::timesync::SenderCapture capture;
    capture.valid = sender_valid;
    capture.realtime_ns = sender_realtime_ns;
    capture.mapping_uncertainty_ns = upstream_uncertainty_ns;

    isaaccapture::timesync::LocalClockObservation observation;
    int64_t now_ns = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        observation = m_tracker.observe(isaaccapture::timesync::sample_local_clock(m_clocks));
        now_ns = m_clocks.monotonic_ns();
    }
    if (observation.stepped)
    {
        // The frame that revealed a realtime step on this host makes no claim.
        isaaccapture::timesync::Qualification refused;
        refused.verdict = isaaccapture::timesync::Verdict::NoLocalClock;
        refused.generation = observation.generation;
        return refused;
    }
    return isaaccapture::timesync::qualify_capture(capture, m_sender_monitor.snapshot().evidence,
                                                   m_pc_monitor.snapshot().evidence, observation, now_ns,
                                                   m_qualifier_config);
}

} // namespace camera_bridge
