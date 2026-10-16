// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_trackers/ego_frame_metadata_tracker.hpp"

#include <set>
#include <stdexcept>
#include <string>

namespace core
{

EgoFrameMetadataTracker::EgoFrameMetadataTracker(const std::string& collection_prefix,
                                                 const std::vector<EgoCameraStream>& streams,
                                                 size_t max_flatbuffer_size)
    : collection_prefix_(collection_prefix), streams_(streams), max_flatbuffer_size_(max_flatbuffer_size)
{
    if (collection_prefix_.empty())
        throw std::invalid_argument("EgoFrameMetadataTracker: collection_prefix is required");
    if (streams_.empty())
        throw std::invalid_argument("EgoFrameMetadataTracker: at least one stream is required");
    if (max_flatbuffer_size_ == 0)
        throw std::invalid_argument("EgoFrameMetadataTracker: max_flatbuffer_size must be positive");

    std::set<EgoCameraStream> unique_streams;
    for (const auto stream : streams_)
    {
        const char* name = EnumNameEgoCameraStream(stream);
        if (name == nullptr || *name == '\0')
        {
            throw std::invalid_argument("EgoFrameMetadataTracker: invalid stream value " +
                                        std::to_string(static_cast<int>(stream)));
        }
        if (!unique_streams.insert(stream).second)
        {
            throw std::invalid_argument("EgoFrameMetadataTracker: duplicate stream " + std::string(name));
        }
        stream_names_.emplace_back(name);
    }
}

const Serialized<EgoFrameMetadata>& EgoFrameMetadataTracker::get_stream_data(const ITrackerSession& session,
                                                                             size_t stream_index) const
{
    return static_cast<const IEgoFrameMetadataTrackerImpl&>(session.get_tracker_impl(*this)).get_stream_data(stream_index);
}

} // namespace core
