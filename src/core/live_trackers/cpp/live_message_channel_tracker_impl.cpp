// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_message_channel_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <schema/message_channel_bfbs_generated.h>
#include <schema/timestamp_generated.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace core
{

std::unique_ptr<MessageChannelMcapChannels> LiveMessageChannelTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                                std::string_view base_name)
{
    return std::make_unique<MessageChannelMcapChannels>(
        writer, base_name,
        std::vector<std::string>(
            MessageChannelRecordingTraits::channels.begin(), MessageChannelRecordingTraits::channels.end()));
}

LiveMessageChannelTrackerImpl::LiveMessageChannelTrackerImpl(const OpenXRSessionHandles& handles,
                                                             const MessageChannelTracker* tracker,
                                                             std::unique_ptr<MessageChannelMcapChannels> mcap_channels)
    : handles_(handles), tracker_(tracker), time_converter_(handles), mcap_channels_(std::move(mcap_channels))
{
    if (handles_.instance == XR_NULL_HANDLE)
        throw std::invalid_argument("LiveMessageChannelTrackerImpl: handles_.instance is XR_NULL_HANDLE");
    if (handles_.session == XR_NULL_HANDLE)
        throw std::invalid_argument("LiveMessageChannelTrackerImpl: handles_.session is XR_NULL_HANDLE");
    if (!handles_.xrGetInstanceProcAddr)
        throw std::invalid_argument("LiveMessageChannelTrackerImpl: handles_.xrGetInstanceProcAddr is null");
    if (!tracker_)
        throw std::invalid_argument("LiveMessageChannelTrackerImpl: tracker_ is null");

    receive_buffer_.resize(tracker_->max_message_size(), 0);

    initialize_functions();

    system_id_ = resolve_system_id();
    channel_uuid_ = make_uuid(tracker_->channel_uuid());
    create_channel();
}

LiveMessageChannelTrackerImpl::~LiveMessageChannelTrackerImpl()
{
    destroy_channel();
}

void LiveMessageChannelTrackerImpl::update(int64_t monotonic_time_ns)
{
    last_update_time_ = monotonic_time_ns;
    const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(monotonic_time_ns);

    MessageChannelMessagesTrackedT native;

    try
    {
        const MessageChannelStatus status = query_status();
        if (status == MessageChannelStatus::DISCONNECTED)
        {
            // Runtime/client disconnected: rebuild the channel object so it can reconnect.
            try_reopen_channel();
        }
        else if (status == MessageChannelStatus::CONNECTED)
        {
            drain_messages(native);
        }
        // For other statuses (CONNECTING / SHUTTING / UNKNOWN), no messages
        // are drained but the sentinel write below still advances the
        // replay frame clock.
    }
    catch (...)
    {
        // Publish what was drained before the failure -- those messages came off the
        // channel and are gone from it. Publishing is also what keeps them from being
        // delivered twice: without it the handle still holds last frame's batch, and the
        // next update assembles into a fresh local and re-encodes it.
        messages_ = pack<MessageChannelMessagesTracked>(native);
        throw;
    }

    // Always encode, including for an empty drain: unlike the single-payload trackers,
    // `data` here is a list, and "no messages this frame" is an empty batch rather than
    // an absent one.
    messages_ = pack<MessageChannelMessagesTracked>(native);

    if (mcap_channels_)
    {
        // The message channel is the replay impl's own frame clock:
        // ReplayMessageChannelTrackerImpl consumes one timestamp-group
        // per session.update() to stay aligned with the rest of the
        // recorded data, so every live update must persist at least one
        // record. When nothing was drained, write a data-null sentinel
        // -- skipping the write would let a connection hiccup desync
        // the replay from the per-frame trackers (head / hand / ...)
        // by the duration of the gap.
        DeviceDataTimestamp timestamp(last_update_time_, last_update_time_, xr_time);
        if (native.data.empty())
        {
            mcap_channels_->write(0, pack_record<MessageChannelMessagesRecord>(nullptr, timestamp));
        }
        else
        {
            for (const auto& msg : native.data)
            {
                mcap_channels_->write(0, pack_record<MessageChannelMessagesRecord>(msg.get(), timestamp));
            }
        }
    }
}

void LiveMessageChannelTrackerImpl::drain_messages(MessageChannelMessagesTrackedT& native)
{
    while (true)
    {
        // First call: query pending byte count without reading.
        uint32_t count_out = 0;
        XrResult query_result = receive_fn_(channel_, 0, &count_out, nullptr);
        if (query_result != XR_SUCCESS)
        {
            if (query_result == XR_ERROR_CHANNEL_NOT_CONNECTED_NV)
                return;
            if (query_result == XR_ERROR_INSTANCE_LOST)
            {
                instance_lost_ = true;
                channel_ = XR_NULL_HANDLE;
                return;
            }
            throw std::runtime_error("LiveMessageChannelTrackerImpl: xrReceiveOpaqueDataChannelNV (query) failed, result=" +
                                     std::to_string(query_result));
        }

        if (count_out == 0)
        {
            break;
        }

        if (count_out > tracker_->max_message_size())
        {
            // Drain the oversized message to unblock the queue, but discard the data.
            // Read in bounded chunks using the pre-allocated receive_buffer_ to avoid
            // allocating a buffer sized by the untrusted remote-supplied count_out.
            logger_->warn("Dropping oversized message ({} bytes, max {})", count_out, tracker_->max_message_size());
            uint32_t remaining = count_out;
            while (remaining > 0)
            {
                uint32_t chunk = std::min(remaining, static_cast<uint32_t>(receive_buffer_.size()));
                uint32_t drained = 0;
                XrResult drain_result = receive_fn_(channel_, chunk, &drained, receive_buffer_.data());
                if (drain_result != XR_SUCCESS || drained == 0)
                {
                    break;
                }
                remaining -= drained;
            }
            if (remaining > 0)
            {
                // Drain failed; stop processing to avoid re-querying the same oversized message.
                break;
            }
            continue;
        }

        // Second call: read the message into the pre-allocated buffer.
        uint32_t read_count = 0;
        XrResult recv_result = receive_fn_(channel_, count_out, &read_count, receive_buffer_.data());
        if (recv_result != XR_SUCCESS)
        {
            if (recv_result == XR_ERROR_CHANNEL_NOT_CONNECTED_NV)
                return;
            if (recv_result == XR_ERROR_INSTANCE_LOST)
            {
                instance_lost_ = true;
                channel_ = XR_NULL_HANDLE;
                return;
            }
            throw std::runtime_error("LiveMessageChannelTrackerImpl: xrReceiveOpaqueDataChannelNV (read) failed, result=" +
                                     std::to_string(recv_result));
        }

        auto message = std::make_shared<MessageChannelMessagesT>();
        message->payload.assign(receive_buffer_.begin(), receive_buffer_.begin() + read_count);
        native.data.push_back(message);
    }
}

MessageChannelStatus LiveMessageChannelTrackerImpl::get_status() const
{
    return query_status();
}

const Serialized<MessageChannelMessagesTracked>& LiveMessageChannelTrackerImpl::get_messages() const
{
    return messages_;
}

void LiveMessageChannelTrackerImpl::send_message(const std::vector<uint8_t>& payload) const
{
    if (channel_ == XR_NULL_HANDLE)
    {
        throw std::runtime_error("LiveMessageChannelTrackerImpl::send_message: channel is not open");
    }

    if (payload.size() > tracker_->max_message_size())
    {
        throw std::runtime_error("LiveMessageChannelTrackerImpl::send_message: payload size " +
                                 std::to_string(payload.size()) + " exceeds max_message_size " +
                                 std::to_string(tracker_->max_message_size()));
    }

    XrOpaqueDataChannelStateNV channel_state{ XR_TYPE_OPAQUE_DATA_CHANNEL_STATE_NV };
    XrResult state_result = get_state_fn_(channel_, &channel_state);
    if (state_result != XR_SUCCESS)
    {
        if (state_result == XR_ERROR_INSTANCE_LOST)
        {
            instance_lost_ = true;
            channel_ = XR_NULL_HANDLE;
        }
        throw std::runtime_error("LiveMessageChannelTrackerImpl: xrGetOpaqueDataChannelStateNV failed, result=" +
                                 std::to_string(state_result));
    }
    if (channel_state.state != XR_OPAQUE_DATA_CHANNEL_STATUS_CONNECTED_NV)
    {
        throw std::runtime_error("LiveMessageChannelTrackerImpl::send_message: channel is not connected");
    }

    const uint8_t* payload_ptr = payload.empty() ? nullptr : payload.data();
    XrResult send_result = send_fn_(channel_, static_cast<uint32_t>(payload.size()), payload_ptr);
    if (send_result != XR_SUCCESS)
    {
        if (send_result == XR_ERROR_INSTANCE_LOST)
        {
            instance_lost_ = true;
            channel_ = XR_NULL_HANDLE;
        }
        throw std::runtime_error("LiveMessageChannelTrackerImpl: xrSendOpaqueDataChannelNV failed, result=" +
                                 std::to_string(send_result));
    }
}

void LiveMessageChannelTrackerImpl::initialize_functions()
{
    loadExtensionFunction(handles_.instance, handles_.xrGetInstanceProcAddr, "xrCreateOpaqueDataChannelNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&create_channel_fn_));
    loadExtensionFunction(handles_.instance, handles_.xrGetInstanceProcAddr, "xrDestroyOpaqueDataChannelNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&destroy_channel_fn_));
    loadExtensionFunction(handles_.instance, handles_.xrGetInstanceProcAddr, "xrGetOpaqueDataChannelStateNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&get_state_fn_));
    loadExtensionFunction(handles_.instance, handles_.xrGetInstanceProcAddr, "xrSendOpaqueDataChannelNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&send_fn_));
    loadExtensionFunction(handles_.instance, handles_.xrGetInstanceProcAddr, "xrReceiveOpaqueDataChannelNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&receive_fn_));
    loadExtensionFunction(handles_.instance, handles_.xrGetInstanceProcAddr, "xrShutdownOpaqueDataChannelNV",
                          reinterpret_cast<PFN_xrVoidFunction*>(&shutdown_fn_));
    loadExtensionFunction(handles_.instance, handles_.xrGetInstanceProcAddr, "xrGetSystem",
                          reinterpret_cast<PFN_xrVoidFunction*>(&get_system_fn_));
}

XrSystemId LiveMessageChannelTrackerImpl::resolve_system_id() const
{
    XrSystemGetInfo get_info{ XR_TYPE_SYSTEM_GET_INFO };
    get_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    XrSystemId system_id = XR_NULL_SYSTEM_ID;
    XrResult result = get_system_fn_(handles_.instance, &get_info, &system_id);
    if (result != XR_SUCCESS)
    {
        throw std::runtime_error("LiveMessageChannelTrackerImpl: xrGetSystem failed, result=" + std::to_string(result));
    }
    return system_id;
}

void LiveMessageChannelTrackerImpl::create_channel()
{
    XrOpaqueDataChannelCreateInfoNV create_info{ XR_TYPE_OPAQUE_DATA_CHANNEL_CREATE_INFO_NV };
    create_info.systemId = system_id_;
    create_info.uuid = channel_uuid_;

    XrResult result = create_channel_fn_(handles_.instance, &create_info, &channel_);
    if (result != XR_SUCCESS)
    {
        // INSTANCE_LOST and RUNTIME_FAILURE both indicate the IPC connection
        // to the runtime is gone; suppress further reopen attempts.
        if (result == XR_ERROR_INSTANCE_LOST || result == XR_ERROR_RUNTIME_FAILURE)
            instance_lost_ = true;
        throw std::runtime_error("LiveMessageChannelTrackerImpl: xrCreateOpaqueDataChannelNV failed, result=" +
                                 std::to_string(result));
    }
}

void LiveMessageChannelTrackerImpl::destroy_channel() noexcept
{
    if (channel_ == XR_NULL_HANDLE)
        return;

    if (!instance_lost_)
    {
        if (shutdown_fn_)
        {
            XrResult result = shutdown_fn_(channel_);
            if (result == XR_ERROR_INSTANCE_LOST || result == XR_ERROR_RUNTIME_FAILURE)
            {
                instance_lost_ = true;
                channel_ = XR_NULL_HANDLE;
                return;
            }
            if (result != XR_SUCCESS)
                logger_->warn("xrShutdownOpaqueDataChannelNV failed, result={}", static_cast<int>(result));
        }
        if (destroy_channel_fn_)
        {
            XrResult result = destroy_channel_fn_(channel_);
            if (result == XR_ERROR_INSTANCE_LOST || result == XR_ERROR_RUNTIME_FAILURE)
                instance_lost_ = true;
            else if (result != XR_SUCCESS)
                logger_->warn("xrDestroyOpaqueDataChannelNV failed, result={}", static_cast<int>(result));
        }
    }
    channel_ = XR_NULL_HANDLE;
}

bool LiveMessageChannelTrackerImpl::try_reopen_channel()
{
    if (instance_lost_)
        return false;
    try
    {
        destroy_channel();
        create_channel();
        return true;
    }
    catch (const std::exception& e)
    {
        logger_->warn("Failed to reopen message channel: {}", e.what());
        return false;
    }
}

MessageChannelStatus LiveMessageChannelTrackerImpl::query_status() const
{
    if (instance_lost_ || channel_ == XR_NULL_HANDLE)
    {
        return MessageChannelStatus::DISCONNECTED;
    }

    XrOpaqueDataChannelStateNV channel_state{ XR_TYPE_OPAQUE_DATA_CHANNEL_STATE_NV };
    XrResult state_result = get_state_fn_(channel_, &channel_state);
    if (state_result != XR_SUCCESS)
    {
        // Only permanently disable the channel on fatal runtime-loss errors;
        // transient failures return DISCONNECTED to allow a reopen attempt.
        if (state_result == XR_ERROR_INSTANCE_LOST || state_result == XR_ERROR_RUNTIME_FAILURE)
            instance_lost_ = true;
        channel_ = XR_NULL_HANDLE;
        return MessageChannelStatus::DISCONNECTED;
    }

    switch (channel_state.state)
    {
    case XR_OPAQUE_DATA_CHANNEL_STATUS_CONNECTING_NV:
        return MessageChannelStatus::CONNECTING;
    case XR_OPAQUE_DATA_CHANNEL_STATUS_CONNECTED_NV:
        return MessageChannelStatus::CONNECTED;
    case XR_OPAQUE_DATA_CHANNEL_STATUS_SHUTTING_NV:
        return MessageChannelStatus::SHUTTING;
    case XR_OPAQUE_DATA_CHANNEL_STATUS_DISCONNECTED_NV:
        return MessageChannelStatus::DISCONNECTED;
    default:
        return MessageChannelStatus::UNKNOWN;
    }
}

XrUuidEXT LiveMessageChannelTrackerImpl::make_uuid(
    const std::array<uint8_t, MessageChannelTracker::CHANNEL_UUID_SIZE>& channel_uuid) const
{
    XrUuidEXT uuid{};
    std::memcpy(uuid.data, channel_uuid.data(), channel_uuid.size());
    return uuid;
}

} // namespace core
