// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_ego_frame_metadata_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <schema/ego_camera_bfbs_generated.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace core
{

namespace
{

std::vector<SchemaTrackerConfig> make_ego_tensor_configs(const EgoFrameMetadataTracker* tracker)
{
    std::vector<SchemaTrackerConfig> configs;
    configs.reserve(tracker->streams().size());
    for (const auto stream : tracker->streams())
    {
        const char* name = EnumNameEgoCameraStream(stream);
        SchemaTrackerConfig config;
        config.collection_id = tracker->collection_prefix() + "/" + name;
        config.max_flatbuffer_size = tracker->max_flatbuffer_size();
        config.tensor_identifier = "frame_metadata";
        config.localized_name = std::string("EgoFrameMetadataTracker_") + name;
        configs.push_back(std::move(config));
    }
    return configs;
}

} // namespace

std::unique_ptr<EgoFrameMetadataMcapChannels> LiveEgoFrameMetadataTrackerImpl::create_mcap_channels(
    mcap::McapWriter& writer, std::string_view base_name, const EgoFrameMetadataTracker* tracker)
{
    return std::make_unique<EgoFrameMetadataMcapChannels>(writer, base_name, tracker->get_stream_names());
}

LiveEgoFrameMetadataTrackerImpl::LiveEgoFrameMetadataTrackerImpl(const OpenXRSessionHandles& handles,
                                                                 const EgoFrameMetadataTracker* tracker,
                                                                 std::unique_ptr<EgoFrameMetadataMcapChannels> mcap_channels)
    : mcap_channels_(std::move(mcap_channels))
{
    auto configs = make_ego_tensor_configs(tracker);
    for (auto& config : configs)
    {
        StreamState state;
        state.reader = std::make_unique<EgoFrameMetadataSchemaTracker>(
            handles, std::move(config), mcap_channels_.get(), streams_.size());
        streams_.push_back(std::move(state));
    }
}

void LiveEgoFrameMetadataTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    for (auto& stream : streams_)
        stream.reader->update(stream.tracked);
}

const Serialized<EgoFrameMetadata>& LiveEgoFrameMetadataTrackerImpl::get_stream_data(size_t stream_index) const
{
    if (stream_index >= streams_.size())
    {
        throw std::out_of_range("EgoFrameMetadataTracker: invalid stream index " + std::to_string(stream_index));
    }
    return streams_[stream_index].tracked;
}

} // namespace core
