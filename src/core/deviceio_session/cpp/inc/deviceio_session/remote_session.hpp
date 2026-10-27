// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/tracker.hpp>
#include <log_bridge/logger.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core
{

class TrackerSampleStreamReader;

/**
 * @brief Configuration for RemoteSession.
 *
 * tracker_names maps each ITracker pointer to the base name its records carry on the stream;
 * the session recording them on the other end must use the same names.
 */
struct RemoteSessionConfig
{
    std::shared_ptr<TrackerSampleStreamReader> stream;
    std::vector<std::pair<const ITracker*, std::string>> tracker_names;
    //! update() publishes the newest tick's samples only while their tick age is at most this.
    int64_t max_tick_age_ns = 250'000'000;
};

// Remote session -- reads tracker data from a sample stream that a session on another machine
// records: the newest sample per tracker, with no file and no history. The samples arrive in the
// MCAP record format, so the replay impls decode them. Whatever receives the stream feeds
// `stream`; this session only reads it.
class RemoteSession : public ITrackerSession
{
public:
    /**
     * @brief Creates an impl per tracker, each subscribing to its topics on the stream.
     * @throws std::invalid_argument, before any impl subscribes, for a missing stream, a null tracker, a
     *         MessageChannelTracker, whose per-message records a stream that keeps only the newest record
     *         cannot carry, or a tracker with metadata (e.g. TactileTracker), which the replay factory reads
     *         before any sample arrives.
     */
    static std::unique_ptr<RemoteSession> run(RemoteSessionConfig config);

    /**
     * @brief Publishes the stream's newest samples to every tracker. Never waits for data.
     *
     * When the newest tick age exceeds max_tick_age_ns (stale), every tracker reports no data
     * until fresh samples arrive.
     */
    void update();

    //! Whether the last update() published fresh samples.
    bool is_fresh() const
    {
        return fresh_;
    }

    //! Tick age of the samples the last update() published, or nullopt when it published none. Slightly
    //! negative when the clock offset estimate is off by more than the transit time.
    std::optional<int64_t> tick_age_ns() const
    {
        return tick_age_ns_;
    }

    const ITrackerImpl& get_tracker_impl(const ITracker& tracker) const override
    {
        auto it = tracker_impls_.find(&tracker);
        if (it == tracker_impls_.end())
        {
            throw std::runtime_error("Tracker implementation not found for tracker: " + std::string(tracker.get_name()));
        }
        return *(it->second);
    }

private:
    explicit RemoteSession(RemoteSessionConfig config);

    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.RemoteSession");
    // Declared before the impls, which read it, so it is destroyed after them.
    std::shared_ptr<TrackerSampleStreamReader> stream_;
    int64_t max_tick_age_ns_;
    bool fresh_ = false;
    std::optional<int64_t> tick_age_ns_;
    std::unordered_map<const ITracker*, std::unique_ptr<ITrackerImpl>> tracker_impls_;
};

} // namespace core
