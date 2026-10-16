// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_ego_imu_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <schema/ego_imu_bfbs_generated.h>

#include <stdexcept>
#include <utility>

namespace core
{
namespace
{
SchemaTrackerConfig config(std::string collection_id, size_t max_size, const char* tensor, const char* name)
{
    return { .collection_id = std::move(collection_id),
             .max_flatbuffer_size = max_size,
             .tensor_identifier = tensor,
             .localized_name = name };
}
} // namespace

std::unique_ptr<EgoImuMcapChannels> LiveEgoImuTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                std::string_view base_name,
                                                                                const EgoImuTracker* tracker)
{
    return std::make_unique<EgoImuMcapChannels>(writer, base_name, tracker->stream_names());
}

LiveEgoImuTrackerImpl::LiveEgoImuTrackerImpl(const OpenXRSessionHandles& handles,
                                             const EgoImuTracker* tracker,
                                             std::unique_ptr<EgoImuMcapChannels> channels)
    : channels_(std::move(channels))
{
    for (const auto& name : tracker->stream_names())
    {
        StreamState state;
        state.reader = std::make_unique<SchemaTracker<EgoImuBatchRecord, EgoImuBatch>>(
            handles,
            config(tracker->collection_prefix() + "/" + name, tracker->max_flatbuffer_size(), "imu_batch",
                   "EGO IMU batch"),
            channels_.get(), streams_.size());
        streams_.push_back(std::move(state));
    }
}

void LiveEgoImuTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    for (auto& stream : streams_)
        stream.reader->update(stream.tracked);
}

const Serialized<EgoImuBatch>& LiveEgoImuTrackerImpl::get_stream_data(size_t stream_index) const
{
    if (stream_index >= streams_.size())
        throw std::out_of_range("EgoImuTracker: invalid stream index");
    return streams_[stream_index].tracked;
}

} // namespace core
