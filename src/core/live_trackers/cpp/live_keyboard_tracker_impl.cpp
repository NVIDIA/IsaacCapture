// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_keyboard_tracker_impl.hpp"

#include <deviceio_trackers/keyboard_tracker.hpp>
#include <mcap/recording_traits.hpp>
#include <schema/keyboard_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <utility>

namespace core
{

std::unique_ptr<KeyboardMcapChannels> LiveKeyboardTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                    std::string_view base_name)
{
    return std::make_unique<KeyboardMcapChannels>(
        writer, base_name,
        std::vector<std::string>(
            KeyboardRecordingTraits::recording_channels.begin(), KeyboardRecordingTraits::recording_channels.end()));
}

LiveKeyboardTrackerImpl::LiveKeyboardTrackerImpl(std::shared_ptr<KeyboardInputState> state,
                                                 std::unique_ptr<KeyboardMcapChannels> mcap_channels)
    : state_(std::move(state)), mcap_channels_(std::move(mcap_channels))
{
    // The session samples from here on: transitions logged before it started are not its input.
    state_->start_session();
}

void LiveKeyboardTrackerImpl::update(int64_t monotonic_time_ns)
{
    // Invalidate first, publish last: the encode below is the only writer.
    tracked_.reset();

    KeyboardInputState::Snapshot snapshot = state_->drain();

    // Keys are captured on the host clock, so every timestamp field is the monotonic time.
    const DeviceDataTimestamp timestamp(monotonic_time_ns, monotonic_time_ns, monotonic_time_ns);

    // No provider and nothing to report: the device is absent this frame. A frame in which the
    // last provider closed still publishes, so its releases reach consumers.
    if (snapshot.provider_count == 0 && snapshot.events.empty())
    {
        tracked_ = publish_and_record<KeyboardOutputRecord>(mcap_channels_.get(), 0, timestamp, nullptr);
        return;
    }

    KeyboardOutputT native;
    native.pressed_keys = std::move(snapshot.pressed_keys);
    native.events.reserve(snapshot.events.size());
    for (const auto& event : snapshot.events)
    {
        native.events.emplace_back(event.timestamp_ns, event.code, event.pressed ? KeyAction_Press : KeyAction_Release);
    }

    tracked_ = publish_and_record(mcap_channels_.get(), 0, timestamp, &native);
}

const Serialized<KeyboardOutput>& LiveKeyboardTrackerImpl::get_data() const
{
    return tracked_;
}

} // namespace core
