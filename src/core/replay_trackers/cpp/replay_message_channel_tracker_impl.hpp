// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/message_channel_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/message_channel_generated.h>

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace core
{

using MessageChannelMcapViewers = McapTrackerViewers<MessageChannelMessagesRecord>;

/**
 * @brief Frame-aligned replay of the `_teleop_control` message channel.
 *
 * The live recorder writes at least one MCAP record per
 * ``session.update()`` call: one record per drained opaque-channel
 * message, or a single data-null sentinel when nothing was drained
 * that update (see ``LiveMessageChannelTrackerImpl::update``). All
 * records produced by a single live update share that update's
 * monotonic-ns timestamp. The message channel is therefore its own
 * frame clock -- the recorded record stream advances by exactly one
 * timestamp-group per live ``session.update()`` call, independent of
 * the live recording's fps or fps variance.
 *
 * To stay aligned with the rest of the replayed trackers (which also
 * emit one record per ``session.update()``), this impl consumes one
 * timestamp-group per replay update: read the first pending record,
 * then keep consuming records as long as their timestamp matches.
 * Records with non-null ``data`` are surfaced via ``get_messages()``;
 * sentinels are silently dropped. The next record (belonging to the
 * following frame) is buffered in ``pending_record_`` for the next
 * ``update()`` call.
 *
 * No average-dt calculation, reference-channel scan, or wall-clock
 * comparison is involved. Replay-loop dt is irrelevant -- N replay
 * updates consume exactly the first N recorded frames, regardless of
 * how many or how few messages were drained on each one.
 */
class ReplayMessageChannelTrackerImpl : public IMessageChannelTrackerImpl
{
public:
    ReplayMessageChannelTrackerImpl(TrackerSampleSource source, std::string_view base_name);

    ReplayMessageChannelTrackerImpl(const ReplayMessageChannelTrackerImpl&) = delete;
    ReplayMessageChannelTrackerImpl& operator=(const ReplayMessageChannelTrackerImpl&) = delete;
    ReplayMessageChannelTrackerImpl(ReplayMessageChannelTrackerImpl&&) = delete;
    ReplayMessageChannelTrackerImpl& operator=(ReplayMessageChannelTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    MessageChannelStatus get_status() const override;
    const Serialized<MessageChannelMessagesTracked>& get_messages() const override;
    void send_message(const std::vector<uint8_t>& payload) const override;

private:
    // The frame's batch, encoded from a builder local to update().
    Serialized<MessageChannelMessagesTracked> messages_;
    std::unique_ptr<MessageChannelMcapViewers> mcap_viewers_;
    // Holds the first record of the next frame, peeked but not yet
    // consumed. McapTrackerViewers::read() advances the underlying
    // LinearMessageView, so detecting a timestamp boundary requires
    // buffering one record across update() calls.
    Serialized<MessageChannelMessagesRecord> pending_record_;
    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace core
