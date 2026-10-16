// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/ego_frame_metadata_tracker_base.hpp>
#include <schema/ego_camera_generated.h>
#include <schema/ego_limits.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace core
{

class EgoFrameMetadataTracker : public ITracker
{
public:
    static constexpr size_t DEFAULT_MAX_FLATBUFFER_SIZE = EGO_MAX_FLATBUFFER_SIZE;

    EgoFrameMetadataTracker(const std::string& collection_prefix,
                            const std::vector<EgoCameraStream>& streams,
                            size_t max_flatbuffer_size = DEFAULT_MAX_FLATBUFFER_SIZE);

    std::string_view get_name() const override
    {
        return "EgoFrameMetadataTracker";
    }

    const Serialized<EgoFrameMetadata>& get_stream_data(const ITrackerSession& session, size_t stream_index) const;

    size_t get_stream_count() const
    {
        return stream_names_.size();
    }

    const std::string& collection_prefix() const
    {
        return collection_prefix_;
    }

    const std::vector<EgoCameraStream>& streams() const
    {
        return streams_;
    }

    size_t max_flatbuffer_size() const
    {
        return max_flatbuffer_size_;
    }

    const std::vector<std::string>& get_stream_names() const
    {
        return stream_names_;
    }

private:
    std::string collection_prefix_;
    std::vector<EgoCameraStream> streams_;
    size_t max_flatbuffer_size_;
    std::vector<std::string> stream_names_;
};

} // namespace core
