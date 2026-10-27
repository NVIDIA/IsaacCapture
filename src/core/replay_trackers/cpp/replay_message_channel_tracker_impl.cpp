// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_message_channel_tracker_impl.hpp"

#include <flatbuffers/flatbuffers.h>
#include <mcap/reader.hpp>
#include <mcap/recording_traits.hpp>
#include <schema/message_channel_bfbs_generated.h>
#include <schema/timestamp_generated.h>

#include <utility>
#include <vector>

namespace core
{
namespace
{

int64_t record_monotonic_ns(const MessageChannelMessagesRecord& record)
{
    // ``timestamp`` is an optional field, so the accessor returns null when the writer
    // dropped it; fall back to 0 so a malformed file does not stall the grouping loop
    // (the consequence is all timestamp-less records collapsing into one synthetic
    // "frame 0", which is the most forgiving behavior for malformed inputs).
    if (record.timestamp() == nullptr)
    {
        return 0;
    }
    return record.timestamp()->available_time_local_common_clock();
}

Serialized<MessageChannelMessagesTracked> finish_batch(
    flatbuffers::FlatBufferBuilder& builder, const std::vector<flatbuffers::Offset<MessageChannelMessages>>& messages)
{
    builder.Finish(CreateMessageChannelMessagesTracked(builder, builder.CreateVector(messages)));
    return Serialized<MessageChannelMessagesTracked>::adopt(builder);
}

} // namespace

ReplayMessageChannelTrackerImpl::ReplayMessageChannelTrackerImpl(TrackerSampleSource source, std::string_view base_name)
    : mcap_viewers_(std::make_unique<MessageChannelMcapViewers>(
          std::move(source),
          base_name,
          std::vector<std::string>(
              MessageChannelRecordingTraits::channels.begin(), MessageChannelRecordingTraits::channels.end()))),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayMessageChannelTrackerImpl"))
{
}

void ReplayMessageChannelTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    // Each update consumes exactly one recorded frame: all records
    // sharing the first pending record's timestamp. See the class
    // docstring for the invariant this relies on (the live recorder
    // writes >=1 record per session.update()).
    //
    // The records stay in wire form and the batch is built straight from their encoded
    // payloads, so each payload is copied once -- into this builder -- rather than into
    // an unpacked record first and out of it again.
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<MessageChannelMessages>> messages;

    try
    {
        if (!pending_record_)
        {
            pending_record_ = mcap_viewers_->read(0);
        }

        if (pending_record_)
        {
            const int64_t frame_ns = record_monotonic_ns(*pending_record_);
            while (pending_record_ && record_monotonic_ns(*pending_record_) == frame_ns)
            {
                // Sentinel records carry no data and only exist to mark a
                // frame boundary; skip them but still advance the iterator so
                // the next update reads the following frame.
                const MessageChannelMessages* message = pending_record_->data();
                if (message != nullptr && message->payload() != nullptr)
                {
                    const auto* payload = message->payload();
                    messages.push_back(
                        CreateMessageChannelMessages(builder, builder.CreateVector(payload->data(), payload->size())));
                }
                pending_record_ = mcap_viewers_->read(0);
            }
        }
    }
    catch (...)
    {
        // Publish the part of the frame that was read -- those records have been consumed
        // from the viewer either way. Publishing is also what keeps them from being
        // delivered twice: without it the handle still holds last frame's batch, and the
        // next update starts from a fresh builder.
        messages_ = finish_batch(builder, messages);
        throw;
    }

    // Always encode, including for an empty batch: `data` is a list here, so "no
    // messages this frame" is an empty batch rather than an absent one.
    messages_ = finish_batch(builder, messages);
}

MessageChannelStatus ReplayMessageChannelTrackerImpl::get_status() const
{
    // No per-frame state is persisted in the MCAP. The channel was clearly
    // connected at record time (otherwise no records would exist); reporting
    // CONNECTED keeps downstream consumers that gate on status happy.
    return MessageChannelStatus::CONNECTED;
}

const Serialized<MessageChannelMessagesTracked>& ReplayMessageChannelTrackerImpl::get_messages() const
{
    return messages_;
}

void ReplayMessageChannelTrackerImpl::send_message(const std::vector<uint8_t>& /*payload*/) const
{
    // Replay has no peer to send to (the live impl writes to
    // xrSendOpaqueDataChannelNV). Log once-per-call and drop the payload --
    // throwing would force every caller to guard their send path, but the
    // operation is genuinely meaningless under replay.
    logger_->warn("send_message: ignored (no peer in replay mode)");
}

} // namespace core
