// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_session/remote_session.hpp"

#include <deviceio_trackers/message_channel_tracker.hpp>
#include <mcap/sample_stream.hpp>
#include <oxr_utils/os_time.hpp>
#include <replay_trackers/replay_deviceio_factory.hpp>

#include <optional>
#include <stdexcept>

namespace core
{

RemoteSession::RemoteSession(RemoteSessionConfig config)
    : stream_(std::move(config.stream)), max_tick_age_ns_(config.max_tick_age_ns)
{
    if (!stream_)
    {
        throw std::invalid_argument("RemoteSession: config has no stream");
    }

    // Checked before any impl subscribes to topics on the shared stream.
    for (const auto& [tracker_ptr, name] : config.tracker_names)
    {
        if (!tracker_ptr)
        {
            throw std::invalid_argument("RemoteSession: tracker '" + name + "' pointer is null in config");
        }
        // Its replay impl reads records until their timestamp changes, which a stream repeating its
        // newest record never does, and a stream keeps one record per topic, not every message.
        if (dynamic_cast<const MessageChannelTracker*>(tracker_ptr) != nullptr)
        {
            throw std::invalid_argument("MessageChannelTracker is not supported over a remote session");
        }
    }

    // The replay impls read through the stream instead of a file.
    ReplayDeviceIOFactory factory(*stream_, config.tracker_names);
    for (const auto& [tracker_ptr, name] : config.tracker_names)
    {
        tracker_impls_.emplace(tracker_ptr, factory.create_tracker_impl(*tracker_ptr));
    }
}

std::unique_ptr<RemoteSession> RemoteSession::run(RemoteSessionConfig config)
{
    isaaccapture::Logger::get("isaaccapture.core.RemoteSession")
        ->info("RemoteSession: Creating remote session with {} trackers", config.tracker_names.size());

    return std::unique_ptr<RemoteSession>(new RemoteSession(std::move(config)));
}

void RemoteSession::update()
{
    const int64_t monotonic_ns = os_monotonic_now_ns();

    const bool fresh = stream_->snapshot(monotonic_ns, max_tick_age_ns_);
    const std::optional<int64_t> published_ns = stream_->published_tick_time_ns();
    tick_age_ns_ = published_ns ? std::optional<int64_t>(monotonic_ns - *published_ns) : std::nullopt;
    if (fresh != fresh_)
    {
        if (fresh)
        {
            logger_->info("RemoteSession: receiving fresh samples");
        }
        else
        {
            logger_->warn("RemoteSession: samples stale (tick age over {} ms); trackers report no data",
                          max_tick_age_ns_ / 1'000'000);
        }
        fresh_ = fresh;
    }

    for (auto& kv : tracker_impls_)
    {
        kv.second->update(monotonic_ns);
    }
}

} // namespace core
