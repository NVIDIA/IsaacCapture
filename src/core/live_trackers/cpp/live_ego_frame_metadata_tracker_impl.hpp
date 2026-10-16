// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/live_trackers/schema_tracker.hpp"

#include <deviceio_trackers/ego_frame_metadata_tracker.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/ego_camera_generated.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

using EgoFrameMetadataMcapChannels = McapTrackerChannels<EgoFrameMetadataRecord>;
using EgoFrameMetadataSchemaTracker = SchemaTracker<EgoFrameMetadataRecord, EgoFrameMetadata>;

class LiveEgoFrameMetadataTrackerImpl : public IEgoFrameMetadataTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return SchemaTrackerBase::get_required_extensions();
    }

    static std::unique_ptr<EgoFrameMetadataMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                              std::string_view base_name,
                                                                              const EgoFrameMetadataTracker* tracker);

    LiveEgoFrameMetadataTrackerImpl(const OpenXRSessionHandles& handles,
                                    const EgoFrameMetadataTracker* tracker,
                                    std::unique_ptr<EgoFrameMetadataMcapChannels> mcap_channels);

    void update(int64_t monotonic_time_ns) override;
    const Serialized<EgoFrameMetadata>& get_stream_data(size_t stream_index) const override;

private:
    struct StreamState
    {
        std::unique_ptr<EgoFrameMetadataSchemaTracker> reader;
        Serialized<EgoFrameMetadata> tracked;
    };

    std::unique_ptr<EgoFrameMetadataMcapChannels> mcap_channels_;
    std::vector<StreamState> streams_;
};

} // namespace core
