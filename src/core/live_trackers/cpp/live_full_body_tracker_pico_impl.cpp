// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_full_body_tracker_pico_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <schema/full_body_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <cassert>
#include <cstring>

namespace core
{

// ============================================================================
// LiveFullBodyTrackerPicoImpl
// ============================================================================

std::unique_ptr<FullBodyMcapChannels> LiveFullBodyTrackerPicoImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                        std::string_view base_name)
{
    return std::make_unique<FullBodyMcapChannels>(
        writer, base_name,
        std::vector<std::string>(
            FullBodyRecordingTraits::recording_channels.begin(), FullBodyRecordingTraits::recording_channels.end()));
}

LiveFullBodyTrackerPicoImpl::LiveFullBodyTrackerPicoImpl(const OpenXRSessionHandles& handles,
                                                         std::unique_ptr<FullBodyMcapChannels> mcap_channels)
    : time_converter_(handles),
      base_space_(handles.space),
      body_tracker_(XR_NULL_HANDLE),
      pfn_create_body_tracker_(nullptr),
      pfn_destroy_body_tracker_(nullptr),
      pfn_locate_body_joints_(nullptr),
      mcap_channels_(std::move(mcap_channels))
{
    auto core_funcs = OpenXRCoreFunctions::load(handles.instance, handles.xrGetInstanceProcAddr);

    XrSystemId system_id;
    XrSystemGetInfo system_info{ XR_TYPE_SYSTEM_GET_INFO };
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    XrResult result = core_funcs.xrGetSystem(handles.instance, &system_info, &system_id);
    if (XR_SUCCEEDED(result))
    {
        XrSystemBodyTrackingPropertiesBD body_tracking_props{ XR_TYPE_SYSTEM_BODY_TRACKING_PROPERTIES_BD };
        XrSystemProperties system_props{ XR_TYPE_SYSTEM_PROPERTIES };
        system_props.next = &body_tracking_props;

        result = core_funcs.xrGetSystemProperties(handles.instance, system_id, &system_props);
        if (XR_FAILED(result))
        {
            throw std::runtime_error("OpenXR: failed to get system properties: " + std::to_string(result));
        }
        if (!body_tracking_props.supportsBodyTracking)
        {
            logger_->warn("Body tracking not supported by this system, running in limp mode");
            return;
        }
    }
    else
    {
        throw std::runtime_error("OpenXR: failed to get system: " + std::to_string(result));
    }

    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrCreateBodyTrackerBD",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_create_body_tracker_));
    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrDestroyBodyTrackerBD",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_destroy_body_tracker_));
    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrLocateBodyJointsBD",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_locate_body_joints_));

    XrBodyTrackerCreateInfoBD create_info{ XR_TYPE_BODY_TRACKER_CREATE_INFO_BD };
    create_info.next = nullptr;
    create_info.jointSet = XR_BODY_JOINT_SET_FULL_BODY_JOINTS_BD;

    result = pfn_create_body_tracker_(handles.session, &create_info, &body_tracker_);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create body tracker: " + std::to_string(result));
    }

    logger_->info("FullBodyTracker initialized (24 joints)");
}

LiveFullBodyTrackerPicoImpl::~LiveFullBodyTrackerPicoImpl()
{
    if (body_tracker_ != XR_NULL_HANDLE)
    {
        assert(pfn_destroy_body_tracker_ != nullptr && "pfn_destroy_body_tracker must not be null");
        pfn_destroy_body_tracker_(body_tracker_);
        body_tracker_ = XR_NULL_HANDLE;
    }
}

void LiveFullBodyTrackerPicoImpl::update(int64_t monotonic_time_ns)
{
    // Invalidate first, publish last: the encode below is the only writer, so no exit path
    // can leave a caller reading last frame's joints.
    tracked_.reset();
    last_update_time_ = monotonic_time_ns;

    if (body_tracker_ == XR_NULL_HANDLE)
    {
        // Policy: limp mode (feature unsupported/unavailable) is non-fatal.
        return;
    }

    const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(monotonic_time_ns);

    XrBodyJointsLocateInfoBD locate_info{ XR_TYPE_BODY_JOINTS_LOCATE_INFO_BD };
    locate_info.next = nullptr;
    locate_info.baseSpace = base_space_;
    locate_info.time = xr_time;

    XrBodyJointLocationBD joint_locations[XR_BODY_JOINT_COUNT_BD];

    XrBodyJointLocationsBD locations{ XR_TYPE_BODY_JOINT_LOCATIONS_BD };
    locations.next = nullptr;
    locations.jointLocationCount = XR_BODY_JOINT_COUNT_BD;
    locations.jointLocations = joint_locations;

    XrResult result = pfn_locate_body_joints_(body_tracker_, &locate_info, &locations);
    if (XR_FAILED(result))
    {
        // A locate call can transiently fail for reasons unrelated to body
        // tracking itself -- e.g. the base/view pose briefly has no valid
        // relation around a client reconnect. This tracker and the Meta one
        // (live_full_body_tracker_meta_impl.cpp) are always polled every frame
        // regardless of which vendor is actually connected, so throwing here
        // would tear down the whole DeviceIO session (see
        // IsaacTeleopClient._get_tracker_data()) over a hiccup in a tracker
        // that may not even be the one currently in use. Degrade to "no data
        // this frame" instead, matching the limp-mode path above.
        std::cerr << "[FullBodyTracker] xrLocateBodyJointsBD failed: " << result << std::endl;
        tracked_.data.reset();
        return;
    }

    FullBodyPoseT data;
    data.all_joint_poses_tracked = locations.allJointPosesTracked;
    data.joints = std::make_shared<BodyJoints>();

    for (uint32_t i = 0; i < XR_BODY_JOINT_COUNT_BD; ++i)
    {
        const auto& joint_loc = joint_locations[i];

        Point position(joint_loc.pose.position.x, joint_loc.pose.position.y, joint_loc.pose.position.z);
        Quaternion orientation(joint_loc.pose.orientation.x, joint_loc.pose.orientation.y, joint_loc.pose.orientation.z,
                               joint_loc.pose.orientation.w);
        Pose pose(position, orientation);

        bool is_valid = (joint_loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                        (joint_loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);

        BodyJointPose joint_pose(pose, is_valid);
        data.joints->mutable_joints()->Mutate(i, joint_pose);
    }

    const DeviceDataTimestamp timestamp(last_update_time_, last_update_time_, xr_time);
    tracked_ = publish_and_record(mcap_channels_.get(), 0, timestamp, &data);
}

const Serialized<FullBodyPose>& LiveFullBodyTrackerPicoImpl::get_body_pose() const
{
    return tracked_;
}

} // namespace core
