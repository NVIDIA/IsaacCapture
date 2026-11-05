// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/live_trackers/schema_tracker.hpp"

#include <deviceio_base/full_body_tracker_base.hpp>
#include <deviceio_base/tracker_vendor.hpp>
#include <mcap/tracker_channels.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/full_body_generated.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace core
{

using FullBodyMcapChannels = McapTrackerChannels<FullBodyPoseRecord>;
using FullBodyJmexSchemaTracker = SchemaTracker<FullBodyPoseRecord, FullBodyPose>;

// Live full-body impl for the "body.jmex" vendor: the j-mex plugin resolves the 24 slots from the
// standard human skeleton MOXI Player has loaded and pushes them on a tensor collection, already
// in OpenXR axes, so this side only reads.
class LiveFullBodyTrackerJmexImpl : public IFullBodyTrackerImpl
{
public:
    static constexpr std::string_view VENDOR_ID = "body.jmex";
    // The plugin's own --collection-id default, used verbatim on both sides.
    static constexpr std::string_view DEFAULT_COLLECTION_ID = "jmex_full_body";
    static constexpr std::string_view TENSOR_IDENTIFIER = "full_body";
    static constexpr size_t MAX_FLATBUFFER_SIZE = 16 * 1024;

    static std::vector<std::string> required_extensions()
    {
        return SchemaTrackerBase::get_required_extensions();
    }

    static void validate_vendor(const TrackerVendor& vendor);

    LiveFullBodyTrackerJmexImpl(const OpenXRSessionHandles& handles,
                                const TrackerVendor& vendor,
                                std::unique_ptr<FullBodyMcapChannels> mcap_channels);

    LiveFullBodyTrackerJmexImpl(const LiveFullBodyTrackerJmexImpl&) = delete;
    LiveFullBodyTrackerJmexImpl& operator=(const LiveFullBodyTrackerJmexImpl&) = delete;
    LiveFullBodyTrackerJmexImpl(LiveFullBodyTrackerJmexImpl&&) = delete;
    LiveFullBodyTrackerJmexImpl& operator=(LiveFullBodyTrackerJmexImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<FullBodyPose>& get_body_pose() const override;

private:
    std::unique_ptr<FullBodyMcapChannels> mcap_channels_;
    FullBodyJmexSchemaTracker schema_reader_;
    Serialized<FullBodyPose> tracked_;
};

} // namespace core
