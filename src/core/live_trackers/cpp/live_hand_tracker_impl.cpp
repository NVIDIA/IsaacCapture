// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_hand_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <schema/hand_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace core
{

namespace
{

template <size_t N>
std::string bounded_string(const char (&value)[N])
{
    const char* end = std::find(value, value + N, '\0');
    return std::string(value, end);
}

std::string ascii_lower(std::string value)
{
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool contains_case_insensitive(const std::string& haystack, const char* needle)
{
    return ascii_lower(haystack).find(needle) != std::string::npos;
}

bool is_display_device_xdev(const XrXDevPropertiesMNDX& properties)
{
    const std::string name = bounded_string(properties.name);
    const std::string serial = bounded_string(properties.serial);

    auto logger = isaaccapture::Logger::get("isaaccapture.core.LiveHandTrackerImpl");
    logger->debug("candidate XDev name: {}, serial: {}", name, serial);

    return contains_case_insensitive(name, "displaydevice") || contains_case_insensitive(name, "display device") ||
           contains_case_insensitive(serial, "displaydevice") || contains_case_insensitive(serial, "display device") ||
           contains_case_insensitive(name, "head device") || contains_case_insensitive(serial, "head device");
}

} // namespace

// ============================================================================
// LiveHandTrackerImpl
// ============================================================================

std::vector<std::string> LiveHandTrackerImpl::required_extensions()
{
    return { XR_EXT_HAND_TRACKING_EXTENSION_NAME, XR_MNDX_XDEV_SPACE_EXTENSION_NAME };
}

std::unique_ptr<HandMcapChannels> LiveHandTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                            std::string_view base_name)
{
    return std::make_unique<HandMcapChannels>(writer, base_name,
                                              std::vector<std::string>(HandRecordingTraits::recording_channels.begin(),
                                                                       HandRecordingTraits::recording_channels.end()));
}

LiveHandTrackerImpl::LiveHandTrackerImpl(const OpenXRSessionHandles& handles,
                                         std::unique_ptr<HandMcapChannels> mcap_channels)
    : time_converter_(handles),
      base_space_(handles.space),
      xdev_list_(XR_NULL_HANDLE),
      pfn_create_hand_tracker_(nullptr),
      pfn_destroy_hand_tracker_(nullptr),
      pfn_locate_hand_joints_(nullptr),
      pfn_create_xdev_list_(nullptr),
      pfn_destroy_xdev_list_(nullptr),
      pfn_enumerate_xdevs_(nullptr),
      pfn_get_xdev_properties_(nullptr),
      mcap_channels_(std::move(mcap_channels))
{
    auto core_funcs = OpenXRCoreFunctions::load(handles.instance, handles.xrGetInstanceProcAddr);

    XrSystemId system_id;
    XrSystemGetInfo system_info{ XR_TYPE_SYSTEM_GET_INFO };
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    XrResult result = core_funcs.xrGetSystem(handles.instance, &system_info, &system_id);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to get OpenXR system: " + std::to_string(result));
    }

    XrSystemHandTrackingPropertiesEXT hand_tracking_props{ XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT };
    XrSystemProperties system_props{ XR_TYPE_SYSTEM_PROPERTIES };
    system_props.next = &hand_tracking_props;

    result = core_funcs.xrGetSystemProperties(handles.instance, system_id, &system_props);
    if (XR_FAILED(result))
    {
        throw std::runtime_error("Failed to get system properties: " + std::to_string(result));
    }
    if (!hand_tracking_props.supportsHandTracking)
    {
        throw std::runtime_error("Hand tracking not supported by this system");
    }

    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrCreateHandTrackerEXT",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_create_hand_tracker_));
    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrDestroyHandTrackerEXT",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_destroy_hand_tracker_));
    loadExtensionFunction(handles.instance, handles.xrGetInstanceProcAddr, "xrLocateHandJointsEXT",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfn_locate_hand_joints_));

    if (!pfn_create_hand_tracker_ || !pfn_destroy_hand_tracker_ || !pfn_locate_hand_joints_)
    {
        throw std::runtime_error("Failed to get hand tracking function pointers");
    }

    try
    {
        initialize_xdev_hand_trackers(handles);

        // Keep the plain session hand trackers wired as the final fallback candidate.
        //
        // Current deployments are expected to use the XDev-backed candidates above because
        // XR_MNDX_xdev_space is now a required extension for live hand tracking. In that
        // normal path these default trackers should not be the data source selected by
        // update_hand(); they sit after every XDev tracker in the priority list and only win
        // if all XDev candidates fail to create or return inactive data.
        //
        // We intentionally keep this path alive instead of deleting it because future runtime
        // modes may use the default/display hand tracker as a real fallback source once the
        // source-priority policy is expanded beyond today's XDev-first behavior.
        try_create_default_hand_tracker(handles.session, XR_HAND_LEFT_EXT, left_hand_trackers_);
        try_create_default_hand_tracker(handles.session, XR_HAND_RIGHT_EXT, right_hand_trackers_);

        if (left_hand_trackers_.empty())
        {
            throw std::runtime_error("Failed to create any left hand tracker");
        }
        if (right_hand_trackers_.empty())
        {
            throw std::runtime_error("Failed to create any right hand tracker");
        }
    }
    catch (...)
    {
        destroy_hand_trackers(left_hand_trackers_);
        destroy_hand_trackers(right_hand_trackers_);
        destroy_xdev_list();
        throw;
    }

    logger_->info("HandTracker initialized (left candidates: {}, right candidates: {})", left_hand_trackers_.size(),
                  right_hand_trackers_.size());
}

LiveHandTrackerImpl::~LiveHandTrackerImpl()
{
    assert(pfn_destroy_hand_tracker_ != nullptr && "pfn_destroy_hand_tracker must not be null");

    destroy_hand_trackers(left_hand_trackers_);
    destroy_hand_trackers(right_hand_trackers_);
    destroy_xdev_list();
}

void LiveHandTrackerImpl::update(int64_t monotonic_time_ns)
{
    // Invalidate first, publish last: the encodes below are the only writers, so no exit
    // path can leave a caller reading last frame's joints.
    left_tracked_.reset();
    right_tracked_.reset();
    last_update_time_ = monotonic_time_ns;
    const XrTime xr_time = time_converter_.convert_monotonic_ns_to_xrtime(monotonic_time_ns);
    std::optional<HandPoseT> left_native;
    std::optional<HandPoseT> right_native;
    update_hand(left_hand_trackers_, xr_time, left_native);
    update_hand(right_hand_trackers_, xr_time, right_native);

    const DeviceDataTimestamp timestamp(last_update_time_, last_update_time_, xr_time);
    left_tracked_ = publish_and_record(mcap_channels_.get(), 0, timestamp, value_ptr(left_native));
    right_tracked_ = publish_and_record(mcap_channels_.get(), 1, timestamp, value_ptr(right_native));
}

const Serialized<HandPose>& LiveHandTrackerImpl::get_left_hand() const
{
    return left_tracked_;
}

const Serialized<HandPose>& LiveHandTrackerImpl::get_right_hand() const
{
    return right_tracked_;
}

void LiveHandTrackerImpl::initialize_xdev_hand_trackers(const OpenXRSessionHandles& handles)
{
    auto load_optional_func = [&handles](const char* name, PFN_xrVoidFunction* ptr) -> bool
    {
        XrResult result = handles.xrGetInstanceProcAddr(handles.instance, name, ptr);
        return XR_SUCCEEDED(result) && *ptr != nullptr;
    };

    if (!load_optional_func("xrCreateXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_create_xdev_list_)) ||
        !load_optional_func("xrDestroyXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_destroy_xdev_list_)) ||
        !load_optional_func("xrEnumerateXDevsMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_enumerate_xdevs_)) ||
        !load_optional_func("xrGetXDevPropertiesMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_get_xdev_properties_)))
    {
        pfn_create_xdev_list_ = nullptr;
        pfn_destroy_xdev_list_ = nullptr;
        pfn_enumerate_xdevs_ = nullptr;
        pfn_get_xdev_properties_ = nullptr;
        return;
    }

    XrCreateXDevListInfoMNDX create_info{ XR_TYPE_CREATE_XDEV_LIST_INFO_MNDX };
    XrResult result = pfn_create_xdev_list_(handles.session, &create_info, &xdev_list_);
    if (XR_FAILED(result))
    {
        xdev_list_ = XR_NULL_HANDLE;
        return;
    }

    uint32_t xdev_count = 0;
    result = pfn_enumerate_xdevs_(xdev_list_, 0, &xdev_count, nullptr);
    if (XR_FAILED(result) || xdev_count == 0)
    {
        destroy_xdev_list();
        return;
    }

    std::vector<XrXDevIdMNDX> xdev_ids(xdev_count);
    result = pfn_enumerate_xdevs_(xdev_list_, xdev_count, &xdev_count, xdev_ids.data());
    if (XR_FAILED(result))
    {
        destroy_xdev_list();
        return;
    }

    std::vector<XrXDevIdMNDX> preferred_xdev_ids;
    std::vector<XrXDevIdMNDX> display_xdev_ids;
    for (const XrXDevIdMNDX xdev_id : xdev_ids)
    {
        XrGetXDevInfoMNDX get_info{ XR_TYPE_GET_XDEV_INFO_MNDX };
        get_info.id = xdev_id;

        XrXDevPropertiesMNDX properties{ XR_TYPE_XDEV_PROPERTIES_MNDX };
        result = pfn_get_xdev_properties_(xdev_list_, &get_info, &properties);
        if (XR_FAILED(result))
        {
            continue;
        }

        if (is_display_device_xdev(properties))
        {
            display_xdev_ids.push_back(xdev_id);
        }
        else
        {
            preferred_xdev_ids.push_back(xdev_id);
        }
    }

    auto add_xdev_candidates = [this, &handles](const std::vector<XrXDevIdMNDX>& xdev_ids_to_add)
    {
        for (const XrXDevIdMNDX xdev_id : xdev_ids_to_add)
        {
            try_create_xdev_hand_tracker(handles.session, xdev_id, XR_HAND_LEFT_EXT, left_hand_trackers_);
            try_create_xdev_hand_tracker(handles.session, xdev_id, XR_HAND_RIGHT_EXT, right_hand_trackers_);
        }
    };

    add_xdev_candidates(preferred_xdev_ids);
    add_xdev_candidates(display_xdev_ids);

    if (left_hand_trackers_.empty() && right_hand_trackers_.empty())
    {
        destroy_xdev_list();
    }
}

bool LiveHandTrackerImpl::try_create_xdev_hand_tracker(XrSession session,
                                                       XrXDevIdMNDX xdev_id,
                                                       XrHandEXT hand,
                                                       std::vector<XrHandTrackerEXT>& trackers)
{
    if (xdev_list_ == XR_NULL_HANDLE || xdev_id == 0)
    {
        return false;
    }

    XrCreateHandTrackerXDevMNDX xdev_create_info{ XR_TYPE_CREATE_HAND_TRACKER_XDEV_MNDX };
    xdev_create_info.xdevList = xdev_list_;
    xdev_create_info.id = xdev_id;

    XrHandTrackerCreateInfoEXT create_info{ XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT };
    create_info.next = &xdev_create_info;
    create_info.hand = hand;
    create_info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;

    // Reserve before creation so push_back cannot throw after OpenXR returns an unowned handle.
    trackers.reserve(trackers.size() + 1);

    XrHandTrackerEXT tracker = XR_NULL_HANDLE;
    const XrResult result = pfn_create_hand_tracker_(session, &create_info, &tracker);
    if (XR_FAILED(result))
    {
        return false;
    }

    trackers.push_back(tracker);
    return true;
}

bool LiveHandTrackerImpl::try_create_default_hand_tracker(XrSession session,
                                                          XrHandEXT hand,
                                                          std::vector<XrHandTrackerEXT>& trackers)
{
    XrHandTrackerCreateInfoEXT create_info{ XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT };
    create_info.hand = hand;
    create_info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;

    // Reserve before creation so push_back cannot throw after OpenXR returns an unowned handle.
    trackers.reserve(trackers.size() + 1);

    XrHandTrackerEXT tracker = XR_NULL_HANDLE;
    const XrResult result = pfn_create_hand_tracker_(session, &create_info, &tracker);
    if (XR_FAILED(result))
    {
        return false;
    }

    trackers.push_back(tracker);
    return true;
}

void LiveHandTrackerImpl::destroy_hand_trackers(std::vector<XrHandTrackerEXT>& trackers)
{
    for (XrHandTrackerEXT& tracker : trackers)
    {
        if (tracker != XR_NULL_HANDLE)
        {
            pfn_destroy_hand_tracker_(tracker);
            tracker = XR_NULL_HANDLE;
        }
    }
    trackers.clear();
}

void LiveHandTrackerImpl::destroy_xdev_list()
{
    if (xdev_list_ != XR_NULL_HANDLE && pfn_destroy_xdev_list_ != nullptr)
    {
        pfn_destroy_xdev_list_(xdev_list_);
        xdev_list_ = XR_NULL_HANDLE;
    }
}

void LiveHandTrackerImpl::update_hand(const std::vector<XrHandTrackerEXT>& trackers,
                                      XrTime time,
                                      std::optional<HandPoseT>& tracked)
{
    for (XrHandTrackerEXT tracker : trackers)
    {
        std::optional<HandPoseT> candidate;
        if (try_update_hand(tracker, time, candidate))
        {
            tracked = std::move(candidate);
            return;
        }
    }

    tracked.reset();
}

bool LiveHandTrackerImpl::try_update_hand(XrHandTrackerEXT tracker, XrTime time, std::optional<HandPoseT>& tracked)
{
    if (tracker == XR_NULL_HANDLE)
    {
        tracked.reset();
        return false;
    }

    XrHandJointsLocateInfoEXT locate_info{ XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT };
    locate_info.baseSpace = base_space_;
    locate_info.time = time;

    XrHandJointLocationEXT joint_locations[XR_HAND_JOINT_COUNT_EXT];

    XrHandJointLocationsEXT locations{ XR_TYPE_HAND_JOINT_LOCATIONS_EXT };
    locations.next = nullptr;
    locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
    locations.jointLocations = joint_locations;

    XrResult result = pfn_locate_hand_joints_(tracker, &locate_info, &locations);
    if (XR_FAILED(result))
    {
        tracked.reset();
        return false;
    }

    if (!locations.isActive)
    {
        // Policy: inactive hand is a common runtime condition; non-fatal.
        tracked.reset();
        return false;
    }

    // Scratch storage for this candidate: update_hand() only commits it once the locate
    // call succeeds, and update() encodes the winner into the published buffer.
    HandPoseT data;
    data.joints = std::make_shared<HandJoints>();

    for (uint32_t i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i)
    {
        const auto& joint_loc = joint_locations[i];

        Point position(joint_loc.pose.position.x, joint_loc.pose.position.y, joint_loc.pose.position.z);
        Quaternion orientation(joint_loc.pose.orientation.x, joint_loc.pose.orientation.y, joint_loc.pose.orientation.z,
                               joint_loc.pose.orientation.w);
        Pose pose(position, orientation);

        bool is_valid = (joint_loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                        (joint_loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);

        HandJointPose joint_pose(pose, is_valid, joint_loc.radius);
        data.joints->mutable_poses()->Mutate(i, joint_pose);
    }

    tracked = std::move(data);
    return true;
}

} // namespace core
