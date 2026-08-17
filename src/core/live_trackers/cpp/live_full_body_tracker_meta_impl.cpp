// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_full_body_tracker_meta_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <schema/full_body_bfbs_generated.h>
#include <schema/timestamp_generated.h>

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>

namespace core
{

namespace
{

// Selects, for each of the 24 canonical BodyJoint slots (schema/fbs/full_body.fbs,
// XR_BD_body_tracking layout), the corresponding joint out of the 84-joint
// XR_META_body_tracking_full_body skeleton. This is the same joint correspondence
// CloudXR.js's PICO_TO_META_JOINT_MAP encoded before the full Quest skeleton was
// sent down the wire; ported here so the reduction happens once, in DeviceIO,
// immediately before the vendor-neutral schema boundary. Positions and raw
// orientations pass through unchanged -- the Quest/PICO orientation-convention
// correction remains a separate step in isaacteleop.retargeting_engine.
constexpr std::array<XrFullBodyJointMETA, XR_BODY_JOINT_COUNT_BD> kBdToMetaJoint = {
    XR_FULL_BODY_JOINT_HIPS_META, // PELVIS
    XR_FULL_BODY_JOINT_LEFT_UPPER_LEG_META, // LEFT_HIP
    XR_FULL_BODY_JOINT_RIGHT_UPPER_LEG_META, // RIGHT_HIP
    XR_FULL_BODY_JOINT_SPINE_LOWER_META, // SPINE1
    XR_FULL_BODY_JOINT_LEFT_LOWER_LEG_META, // LEFT_KNEE
    XR_FULL_BODY_JOINT_RIGHT_LOWER_LEG_META, // RIGHT_KNEE
    XR_FULL_BODY_JOINT_SPINE_MIDDLE_META, // SPINE2
    XR_FULL_BODY_JOINT_LEFT_FOOT_ANKLE_META, // LEFT_ANKLE
    XR_FULL_BODY_JOINT_RIGHT_FOOT_ANKLE_META, // RIGHT_ANKLE
    XR_FULL_BODY_JOINT_SPINE_UPPER_META, // SPINE3
    XR_FULL_BODY_JOINT_LEFT_FOOT_BALL_META, // LEFT_FOOT
    XR_FULL_BODY_JOINT_RIGHT_FOOT_BALL_META, // RIGHT_FOOT
    XR_FULL_BODY_JOINT_NECK_META, // NECK
    XR_FULL_BODY_JOINT_LEFT_SHOULDER_META, // LEFT_COLLAR
    XR_FULL_BODY_JOINT_RIGHT_SHOULDER_META, // RIGHT_COLLAR
    XR_FULL_BODY_JOINT_HEAD_META, // HEAD
    XR_FULL_BODY_JOINT_LEFT_ARM_UPPER_META, // LEFT_SHOULDER
    XR_FULL_BODY_JOINT_RIGHT_ARM_UPPER_META, // RIGHT_SHOULDER
    XR_FULL_BODY_JOINT_LEFT_ARM_LOWER_META, // LEFT_ELBOW
    XR_FULL_BODY_JOINT_RIGHT_ARM_LOWER_META, // RIGHT_ELBOW
    XR_FULL_BODY_JOINT_LEFT_HAND_WRIST_META, // LEFT_WRIST
    XR_FULL_BODY_JOINT_RIGHT_HAND_WRIST_META, // RIGHT_WRIST
    XR_FULL_BODY_JOINT_LEFT_HAND_PALM_META, // LEFT_HAND
    XR_FULL_BODY_JOINT_RIGHT_HAND_PALM_META, // RIGHT_HAND
};

} // namespace

// ============================================================================
// LiveFullBodyTrackerMetaImpl
// ============================================================================

std::unique_ptr<FullBodyMcapChannels> LiveFullBodyTrackerMetaImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                        std::string_view base_name)
{
    return std::make_unique<FullBodyMcapChannels>(
        writer, base_name, FullBodyRecordingTraits::schema_name,
        std::vector<std::string>(
            FullBodyRecordingTraits::recording_channels.begin(), FullBodyRecordingTraits::recording_channels.end()));
}

LiveFullBodyTrackerMetaImpl::LiveFullBodyTrackerMetaImpl(const OpenXRSessionHandles& handles,
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
        XrSystemPropertiesBodyTrackingFullBodyMETA full_body_props{ XR_TYPE_SYSTEM_PROPERTIES_BODY_TRACKING_FULL_BODY_META };
        XrSystemProperties system_props{ XR_TYPE_SYSTEM_PROPERTIES };
        system_props.next = &full_body_props;

        result = core_funcs.xrGetSystemProperties(handles.instance, system_id, &system_props);
        if (XR_FAILED(result))
        {
            throw std::runtime_error("OpenXR: failed to get system properties: " + std::to_string(result));
        }
        if (!full_body_props.supportsFullBodyTracking)
        {
            std::cerr << "[FullBodyTracker] Meta full-body tracking not supported by this system, running in limp mode"
                      << std::endl;
            return;
        }
    }
    else
    {
        throw std::runtime_error("OpenXR: failed to get system: " + std::to_string(result));
    }

    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrCreateBodyTrackerFB",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_create_body_tracker_));
    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrDestroyBodyTrackerFB",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_destroy_body_tracker_));
    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrLocateBodyJointsFB",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_locate_body_joints_));

    XrBodyTrackerCreateInfoFB create_info{ XR_TYPE_BODY_TRACKER_CREATE_INFO_FB };
    create_info.next = nullptr;
    create_info.bodyJointSet = XR_BODY_JOINT_SET_FULL_BODY_META;

    result = pfn_create_body_tracker_(handles.session, &create_info, &body_tracker_);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to create body tracker: " + std::to_string(result));
    }

    std::cout << "FullBodyTracker initialized (84 joints, reduced to 24)" << std::endl;
}

LiveFullBodyTrackerMetaImpl::~LiveFullBodyTrackerMetaImpl()
{
    if (body_tracker_ != XR_NULL_HANDLE)
    {
        assert(pfn_destroy_body_tracker_ != nullptr && "pfn_destroy_body_tracker must not be null");
        pfn_destroy_body_tracker_(body_tracker_);
        body_tracker_ = XR_NULL_HANDLE;
    }
}

void LiveFullBodyTrackerMetaImpl::update(int64_t monotonic_time_ns)
{
    last_update_time_ = monotonic_time_ns;

    if (body_tracker_ == XR_NULL_HANDLE)
    {
        // Policy: limp mode (feature unsupported/unavailable) is non-fatal.
        tracked_.data.reset();
        return;
    }

    const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(monotonic_time_ns);

    XrBodyJointsLocateInfoFB locate_info{ XR_TYPE_BODY_JOINTS_LOCATE_INFO_FB };
    locate_info.next = nullptr;
    locate_info.baseSpace = base_space_;
    locate_info.time = xr_time;

    XrBodyJointLocationFB joint_locations[XR_FULL_BODY_JOINT_COUNT_META];

    XrBodyJointLocationsFB locations{ XR_TYPE_BODY_JOINT_LOCATIONS_FB };
    locations.next = nullptr;
    locations.jointCount = XR_FULL_BODY_JOINT_COUNT_META;
    locations.jointLocations = joint_locations;

    XrResult result = pfn_locate_body_joints_(body_tracker_, &locate_info, &locations);
    if (XR_FAILED(result))
    {
        tracked_.data.reset();
        throw std::runtime_error("[FullBodyTracker] xrLocateBodyJointsFB failed: " + std::to_string(result));
    }

    if (!locations.isActive)
    {
        tracked_.data.reset();
        return;
    }

    // Publish freshly allocated joint storage each frame instead of refilling the previous
    // frame's. The query API hands the pose out by reference and callers may still hold an
    // earlier frame's joints, so an in-place refill would change data already handed out.
    auto data = std::make_shared<FullBodyPoseT>();
    data->joints = std::make_shared<BodyJoints>();

    bool all_tracked = true;
    for (uint32_t bd_index = 0; bd_index < XR_BODY_JOINT_COUNT_BD; ++bd_index)
    {
        const XrBodyJointLocationFB& joint_loc = joint_locations[kBdToMetaJoint[bd_index]];

        Point position(joint_loc.pose.position.x, joint_loc.pose.position.y, joint_loc.pose.position.z);
        Quaternion orientation(joint_loc.pose.orientation.x, joint_loc.pose.orientation.y, joint_loc.pose.orientation.z,
                               joint_loc.pose.orientation.w);
        Pose pose(position, orientation);

        const bool is_valid = (joint_loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                              (joint_loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);
        all_tracked = all_tracked && is_valid;

        BodyJointPose joint_pose(pose, is_valid);
        data->joints->mutable_joints()->Mutate(bd_index, joint_pose);
    }
    data->all_joint_poses_tracked = all_tracked;

    tracked_.data = std::move(data);

    if (mcap_channels_)
    {
        DeviceDataTimestamp timestamp(last_update_time_, last_update_time_, xr_time);
        mcap_channels_->write(0, timestamp, tracked_.data);
    }
}

const FullBodyPoseTrackedT& LiveFullBodyTrackerMetaImpl::get_body_pose() const
{
    return tracked_;
}

} // namespace core
