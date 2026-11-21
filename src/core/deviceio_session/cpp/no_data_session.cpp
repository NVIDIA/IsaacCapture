// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_session/no_data_session.hpp"

#include <deviceio_trackers/message_channel_tracker.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/sample_stream.hpp>
#include <replay_trackers/replay_deviceio_factory.hpp>

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace core
{

namespace
{

//! A message channel with no headset to connect to. It answers as LiveMessageChannelTrackerImpl does while it has
//! no open channel.
class NoDataMessageChannelTrackerImpl : public IMessageChannelTrackerImpl
{
public:
    NoDataMessageChannelTrackerImpl()
    {
        // "No messages" is an empty batch, not an absent one.
        messages_ = pack<MessageChannelMessagesTracked>(MessageChannelMessagesTrackedT{});
    }

    void update(int64_t /*monotonic_time_ns*/) override
    {
    }

    MessageChannelStatus get_status() const override
    {
        return MessageChannelStatus::DISCONNECTED;
    }

    const Serialized<MessageChannelMessagesTracked>& get_messages() const override
    {
        return messages_;
    }

    void send_message(const std::vector<uint8_t>& /*payload*/) const override
    {
        throw std::runtime_error("NoDataSession: MessageChannelTracker::send_message: channel is not open");
    }

private:
    Serialized<MessageChannelMessagesTracked> messages_;
};

bool is_message_channel(const ITracker* tracker)
{
    return dynamic_cast<const MessageChannelTracker*>(tracker) != nullptr;
}

} // namespace

NoDataSession::NoDataSession(const std::vector<const ITracker*>& trackers)
    : stream_(std::make_shared<TrackerSampleStreamReader>())
{
    // The replay impls read the stream; the names only tell their topics apart. A message channel gets an impl
    // of its own: its replay impl reports a connected channel.
    std::vector<std::pair<const ITracker*, std::string>> tracker_names;
    std::unordered_set<const ITracker*> seen;
    for (const ITracker* tracker : trackers)
    {
        if (!tracker)
        {
            throw std::invalid_argument("NoDataSession: tracker pointer is null");
        }
        if (!seen.insert(tracker).second)
        {
            throw std::invalid_argument("NoDataSession: tracker '" + std::string(tracker->get_name()) +
                                        "' is listed twice");
        }
        if (!is_message_channel(tracker))
        {
            tracker_names.emplace_back(tracker, std::to_string(tracker_names.size()));
        }
    }

    ReplayDeviceIOFactory factory(*stream_, tracker_names, /*hold_metadata_trackers=*/true);
    for (const ITracker* tracker : trackers)
    {
        if (is_message_channel(tracker))
        {
            tracker_impls_.emplace(tracker, std::make_unique<NoDataMessageChannelTrackerImpl>());
        }
        else
        {
            tracker_impls_.emplace(tracker, factory.create_tracker_impl(*tracker));
        }
    }
}

std::unique_ptr<NoDataSession> NoDataSession::run(const std::vector<const ITracker*>& trackers)
{
    isaaccapture::Logger::get("isaaccapture.core.NoDataSession")
        ->info("NoDataSession: Creating session with {} trackers that report no data", trackers.size());

    return std::unique_ptr<NoDataSession>(new NoDataSession(trackers));
}

const ITrackerImpl& NoDataSession::get_tracker_impl(const ITracker& tracker) const
{
    auto it = tracker_impls_.find(&tracker);
    if (it == tracker_impls_.end())
    {
        throw std::runtime_error("Tracker implementation not found for tracker: " + std::string(tracker.get_name()));
    }
    return *(it->second);
}

} // namespace core
