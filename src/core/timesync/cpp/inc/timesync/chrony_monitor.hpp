// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "capture_qualifier.hpp"
#include "command_runner.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace isaaccapture::timesync
{

struct ChronyMonitorConfig
{
    std::string chronyc_path = "chronyc";
    /// Empty queries the local chronyd. A host queries a remote one, which must allow this client (cmdallow);
    /// chronyd accepts only monitoring commands from the network.
    std::string host;
    int port = 0; ///< 0 uses chronyc's default.
    std::chrono::milliseconds poll_interval{ 1000 };
    std::chrono::milliseconds command_timeout{ 2000 };
};

enum class MonitorStatus
{
    NotPolled,
    Ok,
    CommandFailed, ///< Could not start, non-zero exit, or truncated output.
    TimedOut,
    ParseFailed,
};

const char* to_string(MonitorStatus s);

struct MonitorSnapshot
{
    /// Newest good report, stamped when its query began on the monitor's monotonic clock. It is kept across
    /// failures so that it simply ages into staleness; check `last_status` for why it stopped updating.
    std::optional<ChronyEvidence> evidence;
    MonitorStatus last_status = MonitorStatus::NotPolled;
    uint64_t polls = 0;
    uint64_t failures = 0;
};

/// Polls `chronyc -c tracking` on a background thread, outside any frame path. One report per poll, no queueing.
class ChronyMonitor
{
public:
    using NowFn = std::function<int64_t()>;

    /// The runner and clocks default to the real process runner, CLOCK_MONOTONIC and CLOCK_REALTIME.
    explicit ChronyMonitor(ChronyMonitorConfig config,
                           CommandRunner runner = {},
                           NowFn now_monotonic_ns = {},
                           NowFn now_realtime_ns = {});
    ~ChronyMonitor();

    ChronyMonitor(const ChronyMonitor&) = delete;
    ChronyMonitor& operator=(const ChronyMonitor&) = delete;

    /// Starts the polling thread; a second call does nothing.
    void start();
    /// Wakes and joins the thread; safe to call repeatedly. Returns within the command timeout at worst.
    void stop();

    /// One poll on the calling thread; this is what the thread runs, and what tests drive directly.
    MonitorStatus poll_once();

    MonitorSnapshot snapshot() const;

private:
    void run();

    ChronyMonitorConfig config_;
    CommandRunner runner_;
    NowFn now_;
    NowFn now_realtime_;
    /// Serialises start() and stop() so a start cannot slip in while a stop is still joining the worker.
    std::mutex lifecycle_mutex_;
    mutable std::mutex mutex_;
    MonitorSnapshot snapshot_;
    std::condition_variable wake_;
    bool stopping_ = false;
    std::thread thread_;
};

} // namespace isaaccapture::timesync
