// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_head_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <schema/head_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace core
{

// ============================================================================
// LiveHeadTrackerImpl
// ============================================================================

std::unique_ptr<HeadMcapChannels> LiveHeadTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                            std::string_view base_name)
{
    return std::make_unique<HeadMcapChannels>(writer, base_name,
                                              std::vector<std::string>(HeadRecordingTraits::recording_channels.begin(),
                                                                       HeadRecordingTraits::recording_channels.end()));
}

LiveHeadTrackerImpl::LiveHeadTrackerImpl(const OpenXRSessionHandles& handles,
                                         std::unique_ptr<HeadMcapChannels> mcap_channels)
    : core_funcs_(OpenXRCoreFunctions::load(handles.instance, handles.xrGetInstanceProcAddr)),
      handles_(handles),
      time_converter_(handles),
      base_space_(handles.space),
      view_space_(createReferenceSpace(core_funcs_,
                                       handles.session,
                                       { .type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
                                         .referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW,
                                         .poseInReferenceSpace = { .orientation = { 0, 0, 0, 1 } } })),
      mcap_channels_(std::move(mcap_channels))
{
}

void LiveHeadTrackerImpl::update(int64_t monotonic_time_ns)
{
    // Invalidate first, publish last: the encode below is the only writer, so no exit path
    // can leave a caller reading last frame's pose.
    tracked_.reset();
    last_update_time_ = monotonic_time_ns;

    if (!handles_.is_running())
    {
        return;
    }

    const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(monotonic_time_ns);

    XrSpaceLocation location{ XR_TYPE_SPACE_LOCATION };
    XrResult result = core_funcs_.xrLocateSpace(view_space_.get(), base_space_, xr_time, &location);

    if (XR_FAILED(result))
    {
        throw std::runtime_error("[HeadTracker] xrLocateSpace failed: " + std::to_string(result));
    }

    // OpenXR: VALID => pose may be read; TRACKED => actively tracked. Keep these
    // separate so is_valid stays OpenXR-faithful (CloudXR may leave VALID set after disconnect).
    const bool position_valid = (location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;
    const bool orientation_valid = (location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0;
    const bool position_tracked = (location.locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT) != 0;
    const bool orientation_tracked = (location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT) != 0;

    HeadPoseT native;

    native.is_valid = position_valid && orientation_valid;
    native.is_tracked = native.is_valid && position_tracked && orientation_tracked;

    if (native.is_valid)
    {
        // Pose is readable whenever VALID (including untracked placeholder / last pose).
        Point position(location.pose.position.x, location.pose.position.y, location.pose.position.z);
        Quaternion orientation(location.pose.orientation.x, location.pose.orientation.y, location.pose.orientation.z,
                               location.pose.orientation.w);
        native.pose = std::make_shared<Pose>(position, orientation);
    }
    else
    {
        // is_valid=false: pose contents are unspecified; leave a default filler.
        native.pose = std::make_shared<Pose>();
    }

    const DeviceDataTimestamp timestamp(last_update_time_, last_update_time_, xr_time);
    tracked_ = publish_and_record(mcap_channels_.get(), 0, timestamp, &native);
}

const Serialized<HeadPose>& LiveHeadTrackerImpl::get_head() const
{
    return tracked_;
}

} // namespace core
