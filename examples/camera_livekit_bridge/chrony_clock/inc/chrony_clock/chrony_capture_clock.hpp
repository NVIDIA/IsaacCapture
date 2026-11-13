// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <timesync/capture_qualifier.hpp>
#include <timesync/chrony_monitor.hpp>
#include <timesync/local_clock.hpp>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

namespace camera_bridge
{

struct ChronyCaptureClockConfig
{
    std::string chronyc_path = "chronyc";
    /// The camera host's chronyd, queried over chronyc's read-only monitoring interface (the camera host must
    /// `cmdallow` this machine). Required: without it the sender's synchronization is unknown and nothing qualifies.
    std::string sender_host;
    int sender_port = 0;
    /// Upper-case hex reference id this machine's chronyd must have selected: the camera host as its NTP source.
    std::string expected_sender_reference_id;
    /// The command port of this machine's chronyd that measures the camera host, queried on the loopback address. 0
    /// queries the system chronyd over its local socket.
    int pc_port = 0;
    int64_t budget_ns = 5'000'000;
    /// After a realtime step on this host no frame is given a timestamp for this long, whatever the evidence says. A
    /// take can only notice the step through a frame that has no timestamp, or through silence; holding timestamps
    /// off for longer than a recording consumer tolerates silence (5 s) means a take cannot continue across a step even
    /// if the media path drops the frames that were refused. Frames still flow without timestamps meanwhile.
    int64_t step_hold_ns = 8'000'000'000;
    std::chrono::milliseconds poll_interval{ 1000 };
    std::chrono::milliseconds command_timeout{ 2000 };
};

/**
 * @brief Qualifies a sender CLOCK_REALTIME capture time against chrony's evidence and moves it onto this host's
 * monotonic clock.
 *
 * It polls this host's chronyd and the camera host's in the background and samples the local realtime/monotonic pair
 * per frame. A frame the evidence cannot vouch for gets a refusal verdict and no time, and the publisher sends it
 * without a timestamp, which a recording consumer treats as a timing fault. A detected realtime step on this host
 * refuses the frame that revealed it and every frame after it until both chronyds have been polled again after the
 * step: evidence read before it described the old clock and is never reused, and for `step_hold_ns` after it no frame
 * is timestamped at all, so the frames in between carry no timestamp and a take in progress ends with that fault even
 * if some of those frames are lost on the way to it. Frames after that qualify against the new clock generation.
 * Thread-safe.
 */
class ChronyCaptureClock
{
public:
    /// The runner and clocks default to the real process runner and the system clocks; tests substitute fakes.
    explicit ChronyCaptureClock(ChronyCaptureClockConfig config,
                                isaaccapture::timesync::CommandRunner runner = {},
                                isaaccapture::timesync::ClockReader clocks = {});
    ~ChronyCaptureClock();

    ChronyCaptureClock(const ChronyCaptureClock&) = delete;
    ChronyCaptureClock& operator=(const ChronyCaptureClock&) = delete;

    void start();
    void stop();

    /// One immediate poll of both monitors on the calling thread, for startup and tests.
    void poll_once();

    isaaccapture::timesync::Qualification qualify(bool sender_valid,
                                                  int64_t sender_realtime_ns,
                                                  int64_t upstream_uncertainty_ns);

private:
    ChronyCaptureClockConfig m_config;
    isaaccapture::timesync::ClockReader m_clocks;
    isaaccapture::timesync::QualifierConfig m_qualifier_config;
    isaaccapture::timesync::ChronyMonitor m_pc_monitor;
    isaaccapture::timesync::ChronyMonitor m_sender_monitor;
    std::mutex m_mutex;
    isaaccapture::timesync::LocalClockTracker m_tracker;
    /// Monotonic time of the latest detected realtime step on this host; chrony evidence read at or before it is not
    /// used.
    int64_t m_evidence_floor_ns = 0;
    /// Monotonic time until which no frame is timestamped after a detected step.
    int64_t m_hold_until_ns = 0;
};

} // namespace camera_bridge
