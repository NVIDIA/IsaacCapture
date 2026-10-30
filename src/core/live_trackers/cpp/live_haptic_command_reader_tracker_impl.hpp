// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/live_trackers/schema_tracker.hpp"

#include <deviceio_base/haptic_command_reader_tracker_base.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/haptic_command_generated.h>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

// MCAP recording disabled for v1; the Record template arg is required by
// SchemaTracker but the impl passes mcap_channels=nullptr.
using HapticCommandSchemaTracker = SchemaTracker<HapticCommandRecord, HapticCommand>;

class LiveHapticCommandReaderTrackerImpl : public IHapticCommandReaderTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return SchemaTrackerBase::get_required_extensions();
    }

    LiveHapticCommandReaderTrackerImpl(const OpenXRSessionHandles& handles, const HapticCommandReaderTracker* tracker);

    LiveHapticCommandReaderTrackerImpl(const LiveHapticCommandReaderTrackerImpl&) = delete;
    LiveHapticCommandReaderTrackerImpl& operator=(const LiveHapticCommandReaderTrackerImpl&) = delete;
    LiveHapticCommandReaderTrackerImpl(LiveHapticCommandReaderTrackerImpl&&) = delete;
    LiveHapticCommandReaderTrackerImpl& operator=(LiveHapticCommandReaderTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<HapticCommand>& get_data() const override;
    const Serialized<HapticCommand>& get_data(std::string_view endpoint) const override;

private:
    HapticCommandSchemaTracker schema_reader_;
    // Latest command per endpoint. One push-tensor collection carries every
    // endpoint's command (each tagged by HapticCommand.endpoint); bucketing by
    // that tag keeps concurrent endpoints (e.g. left/right) from overwriting one
    // another the way a single latest-sample slot would.
    std::map<std::string, Serialized<HapticCommand>, std::less<>> tracked_by_endpoint_;
    // Endpoint of the most recently drained sample, backing the no-arg get_data().
    std::string latest_endpoint_;
};

} // namespace core
