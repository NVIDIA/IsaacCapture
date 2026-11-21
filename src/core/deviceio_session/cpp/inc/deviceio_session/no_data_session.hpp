// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/tracker.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

namespace core
{

class TrackerSampleStreamReader;

// No-data session -- holds a tracker of every kind and has no data for any of them. Every accessor
// returns an empty handle; a message channel reports DISCONNECTED and an empty batch, and refuses to
// send. Nothing feeds it and it has no update(). A LIVE TeleopSession reads it until a headset has
// reached the OpenXR runtime.
class NoDataSession : public ITrackerSession
{
public:
    /**
     * @brief Creates an impl per tracker.
     * @throws std::invalid_argument for a null tracker, a tracker listed twice or a tracker type without an impl.
     */
    static std::unique_ptr<NoDataSession> run(const std::vector<const ITracker*>& trackers);

    const ITrackerImpl& get_tracker_impl(const ITracker& tracker) const override;

private:
    explicit NoDataSession(const std::vector<const ITracker*>& trackers);

    // Never fed. Declared before the impls, which read it, so it is destroyed after them.
    std::shared_ptr<TrackerSampleStreamReader> stream_;
    std::unordered_map<const ITracker*, std::unique_ptr<ITrackerImpl>> tracker_impls_;
};

} // namespace core
