// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/timesync/chrony_monitor.hpp"

#include "inc/timesync/chrony_tracking.hpp"
#include "inc/timesync/local_clock.hpp"

#include <vector>

namespace isaaccapture::timesync
{

const char* to_string(MonitorStatus s)
{
    switch (s)
    {
    case MonitorStatus::NotPolled:
        return "not_polled";
    case MonitorStatus::Ok:
        return "ok";
    case MonitorStatus::CommandFailed:
        return "command_failed";
    case MonitorStatus::TimedOut:
        return "timed_out";
    case MonitorStatus::ParseFailed:
        return "parse_failed";
    }
    return "unknown";
}

ChronyMonitor::ChronyMonitor(ChronyMonitorConfig config, CommandRunner runner, NowFn now_monotonic_ns, NowFn now_realtime_ns)
    : config_(std::move(config)),
      runner_(runner ? std::move(runner) : posix_command_runner()),
      now_(now_monotonic_ns ? std::move(now_monotonic_ns) : system_clock_reader().monotonic_ns),
      now_realtime_(now_realtime_ns ? std::move(now_realtime_ns) : system_clock_reader().realtime_ns)
{
}

ChronyMonitor::~ChronyMonitor()
{
    stop();
}

MonitorStatus ChronyMonitor::poll_once()
{
    std::vector<std::string> argv{ config_.chronyc_path, "-c" };
    if (!config_.host.empty())
    {
        argv.push_back("-h");
        argv.push_back(config_.host);
    }
    if (config_.port != 0)
    {
        argv.push_back("-p");
        argv.push_back(std::to_string(config_.port));
    }
    argv.push_back("tracking");

    const int64_t started_ns = now_();
    const int64_t started_realtime_ns = now_realtime_();
    const CommandResult r = runner_(argv, config_.command_timeout);

    MonitorStatus status = MonitorStatus::Ok;
    std::optional<ChronyEvidence> fresh;
    if (r.timed_out)
    {
        status = MonitorStatus::TimedOut;
    }
    else if (!r.started || r.truncated || r.exit_code != 0)
    {
        status = MonitorStatus::CommandFailed;
    }
    else
    {
        ChronyParseResult parsed = parse_tracking_csv(r.output);
        if (!parsed.tracking)
        {
            status = MonitorStatus::ParseFailed;
        }
        else
        {
            // Stamped when the query began: the report is no newer than that, so its age is never understated.
            fresh = ChronyEvidence{ std::move(*parsed.tracking), started_ns, started_realtime_ns };
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ++snapshot_.polls;
    snapshot_.last_status = status;
    if (fresh)
    {
        snapshot_.evidence = std::move(fresh);
    }
    else
    {
        ++snapshot_.failures;
    }
    return status;
}

MonitorSnapshot ChronyMonitor::snapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

void ChronyMonitor::start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_.joinable())
    {
        return;
    }
    stopping_ = false;
    thread_ = std::thread([this] { run(); });
}

void ChronyMonitor::stop()
{
    std::thread joinable;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        joinable = std::move(thread_);
    }
    wake_.notify_all();
    if (joinable.joinable())
    {
        joinable.join();
    }
}

void ChronyMonitor::run()
{
    for (;;)
    {
        poll_once();
        std::unique_lock<std::mutex> lock(mutex_);
        if (wake_.wait_for(lock, config_.poll_interval, [this] { return stopping_; }))
        {
            return;
        }
    }
}

} // namespace isaaccapture::timesync
