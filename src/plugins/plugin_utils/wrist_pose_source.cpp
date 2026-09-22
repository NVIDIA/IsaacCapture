// SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/plugin_utils/wrist_pose_source.hpp"

#include <oxr_utils/math.hpp>
#include <oxr_utils/pose_conversions.hpp>

#include <algorithm>
#include <iostream>
#include <string>

namespace plugin_utils
{

namespace
{

// Returns true if the OpenXR loader/runtime advertises the given extension.
// xrEnumerateInstanceExtensionProperties is a loader-level function that can be
// called before any XrInstance exists, so this is safe to use at init time.
bool is_openxr_extension_supported(const char* ext_name)
{
    uint32_t count = 0;
    if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr)))
    {
        return false;
    }
    std::vector<XrExtensionProperties> props(count, XrExtensionProperties{ XR_TYPE_EXTENSION_PROPERTIES });
    if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, count, &count, props.data())))
    {
        return false;
    }
    return std::any_of(props.begin(), props.end(),
                       [ext_name](const XrExtensionProperties& p) { return std::string(p.extensionName) == ext_name; });
}

} // namespace

WristPoseSource::Requirements WristPoseSource::collect_requirements(WristSourceMode mode)
{
    Requirements req;

    if (mode != WristSourceMode::Controller)
    {
        // Both extensions are optional: without them the constructor falls
        // back to the controller source (or honest untracked output).
        const bool xdev_supported = is_openxr_extension_supported(XR_MNDX_XDEV_SPACE_EXTENSION_NAME);
        const bool hand_tracking_supported = is_openxr_extension_supported(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
        if (xdev_supported && hand_tracking_supported)
        {
            req.extensions.push_back(XR_MNDX_XDEV_SPACE_EXTENSION_NAME);
            req.extensions.push_back(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
        }
        else
        {
            std::cout << "[WristPoseSource] " << XR_MNDX_XDEV_SPACE_EXTENSION_NAME << " and/or "
                      << XR_EXT_HAND_TRACKING_EXTENSION_NAME << " not supported by the runtime; "
                      << "the optical hand-tracking wrist source will be unavailable." << std::endl;
        }
    }

    if (mode != WristSourceMode::HandTracking)
    {
        req.controller_tracker = std::make_shared<core::ControllerTracker>();
        req.trackers.push_back(req.controller_tracker);
    }

    return req;
}

WristPoseSource::WristPoseSource(const WristSourceConfig& config,
                                 const core::OpenXRSessionHandles& handles,
                                 core::DeviceIOSession* deviceio_session,
                                 std::shared_ptr<core::ControllerTracker> controller_tracker)
    : m_config(config),
      m_handles(handles),
      m_deviceio_session(deviceio_session),
      m_controller_tracker(std::move(controller_tracker))
{
    if (m_config.mode != WristSourceMode::Controller)
    {
        initialize_xdev_hand_trackers();
    }

    const bool controller_available = m_controller_tracker != nullptr && m_deviceio_session != nullptr;
    std::cout << "[WristPoseSource] wrist sources: optical=" << (m_xdev_available ? "available" : "unavailable")
              << " controller=" << (controller_available ? "available" : "unavailable") << std::endl;
}

WristPoseSource::~WristPoseSource()
{
    cleanup_xdev_hand_trackers();
}

WristSample WristPoseSource::query(bool is_left, XrTime time)
{
    HandState& state = is_left ? m_left : m_right;

    XrPosef pose;
    bool tracked = false;
    bool got_pose = false;

    if (m_config.mode != WristSourceMode::Controller && m_xdev_available)
    {
        got_pose = query_xdev(is_left, time, pose, tracked);
    }

    // Keep a valid-but-untracked optical pose instead of switching sources.
    if (!got_pose && m_config.mode != WristSourceMode::HandTracking)
    {
        got_pose = query_controller(is_left, pose, tracked);
    }

    if (got_pose)
    {
        state.last_pose = pose;
        state.has_pose = true;
        return { pose, true, tracked };
    }

    // No source this frame: reuse the last good pose so a brief dropout does
    // not teleport the hand, but never advertise it as tracked.
    return { state.last_pose, state.has_pose, false };
}

void WristPoseSource::initialize_xdev_hand_trackers()
{
    auto load_func = [this](const char* name, PFN_xrVoidFunction* ptr) -> bool
    {
        XrResult result = m_handles.xrGetInstanceProcAddr(m_handles.instance, name, ptr);
        return XR_SUCCEEDED(result) && *ptr != nullptr;
    };

    if (!load_func("xrCreateXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_create_xdev_list)) ||
        !load_func("xrDestroyXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_destroy_xdev_list)) ||
        !load_func("xrEnumerateXDevsMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_enumerate_xdevs)) ||
        !load_func("xrGetXDevPropertiesMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_get_xdev_properties)))
    {
        std::cerr << "[WristPoseSource] XR_MNDX_xdev_space functions unavailable; optical wrist source disabled"
                  << std::endl;
        return;
    }

    if (!load_func("xrCreateHandTrackerEXT", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_create_hand_tracker)) ||
        !load_func("xrDestroyHandTrackerEXT", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_destroy_hand_tracker)) ||
        !load_func("xrLocateHandJointsEXT", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_locate_hand_joints)))
    {
        std::cerr << "[WristPoseSource] XR_EXT_hand_tracking functions unavailable; optical wrist source disabled"
                  << std::endl;
        return;
    }

    XrCreateXDevListInfoMNDX create_info{ XR_TYPE_CREATE_XDEV_LIST_INFO_MNDX };
    XrResult result = m_pfn_create_xdev_list(m_handles.session, &create_info, &m_xdev_list);
    if (XR_FAILED(result))
    {
        std::cerr << "[WristPoseSource] Failed to create XDevList; optical wrist source disabled" << std::endl;
        return;
    }

    uint32_t xdev_count = 0;
    result = m_pfn_enumerate_xdevs(m_xdev_list, 0, &xdev_count, nullptr);
    if (XR_FAILED(result) || xdev_count == 0)
    {
        std::cerr << "[WristPoseSource] No XDevs found; optical wrist source disabled" << std::endl;
        return;
    }

    std::vector<XrXDevIdMNDX> xdev_ids(xdev_count);
    result = m_pfn_enumerate_xdevs(m_xdev_list, xdev_count, &xdev_count, xdev_ids.data());
    if (XR_FAILED(result))
    {
        return;
    }

    // Find native hand tracking devices by matching against their serial strings.
    //
    // NOTE: The serial values "Head Device (0)" (left) and "Head Device (1)" (right) are
    // NOT defined by the XR_MNDX_xdev_space specification. They are an observed runtime-
    // specific naming convention (e.g. Monado). If a runtime changes these display names
    // across firmware or software updates the match below will silently fail.
    // See: https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html (XR_MNDX_xdev_space)
    XrXDevIdMNDX left_xdev_id = 0;
    XrXDevIdMNDX right_xdev_id = 0;
    std::vector<std::string> seen_serials;

    for (const auto& xdev_id : xdev_ids)
    {
        XrGetXDevInfoMNDX get_info{ XR_TYPE_GET_XDEV_INFO_MNDX };
        get_info.id = xdev_id;

        XrXDevPropertiesMNDX properties{ XR_TYPE_XDEV_PROPERTIES_MNDX };
        result = m_pfn_get_xdev_properties(m_xdev_list, &get_info, &properties);
        if (XR_FAILED(result))
        {
            continue;
        }

        std::string serial_str = properties.serial ? properties.serial : "";
        seen_serials.push_back(serial_str);

        if (serial_str == "Head Device (0)")
        {
            left_xdev_id = xdev_id;
        }
        else if (serial_str == "Head Device (1)")
        {
            right_xdev_id = xdev_id;
        }
    }

    if (left_xdev_id == 0 || right_xdev_id == 0)
    {
        std::string serials_list;
        for (const auto& s : seen_serials)
        {
            if (!serials_list.empty())
            {
                serials_list += ", ";
            }
            serials_list += '"';
            serials_list += s;
            serials_list += '"';
        }
        std::cerr << "[WristPoseSource] Could not match optical hand-tracking XDevs by serial. "
                  << "Expected \"Head Device (0)\" (left) and \"Head Device (1)\" (right), "
                  << "but found: [" << serials_list << "]. "
                  << "These serial strings are runtime-specific and may have changed." << std::endl;
    }

    auto create_tracker = [this](XrXDevIdMNDX xdev_id, XrHandEXT hand, XrHandTrackerEXT& out_tracker) -> bool
    {
        if (xdev_id == 0)
        {
            return false;
        }

        XrCreateHandTrackerXDevMNDX xdev_create_info{ XR_TYPE_CREATE_HAND_TRACKER_XDEV_MNDX };
        xdev_create_info.xdevList = m_xdev_list;
        xdev_create_info.id = xdev_id;

        XrHandTrackerCreateInfoEXT create_info{ XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT };
        create_info.next = &xdev_create_info;
        create_info.hand = hand;
        create_info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;

        return XR_SUCCEEDED(m_pfn_create_hand_tracker(m_handles.session, &create_info, &out_tracker));
    };

    const bool left_ok = create_tracker(left_xdev_id, XR_HAND_LEFT_EXT, m_native_left_hand_tracker);
    const bool right_ok = create_tracker(right_xdev_id, XR_HAND_RIGHT_EXT, m_native_right_hand_tracker);

    if (left_ok && right_ok)
    {
        m_xdev_available = true;
    }
    else
    {
        std::cerr << "[WristPoseSource] Failed to create native hand trackers; optical wrist source disabled"
                  << std::endl;
        cleanup_xdev_hand_trackers();
    }
}

void WristPoseSource::cleanup_xdev_hand_trackers()
{
    if (m_native_left_hand_tracker != XR_NULL_HANDLE && m_pfn_destroy_hand_tracker)
    {
        m_pfn_destroy_hand_tracker(m_native_left_hand_tracker);
        m_native_left_hand_tracker = XR_NULL_HANDLE;
    }
    if (m_native_right_hand_tracker != XR_NULL_HANDLE && m_pfn_destroy_hand_tracker)
    {
        m_pfn_destroy_hand_tracker(m_native_right_hand_tracker);
        m_native_right_hand_tracker = XR_NULL_HANDLE;
    }
    if (m_xdev_list != XR_NULL_HANDLE && m_pfn_destroy_xdev_list)
    {
        m_pfn_destroy_xdev_list(m_xdev_list);
        m_xdev_list = XR_NULL_HANDLE;
    }
    m_xdev_available = false;
}

bool WristPoseSource::query_xdev(bool is_left, XrTime time, XrPosef& out_pose, bool& out_tracked)
{
    out_tracked = false;

    const XrHandTrackerEXT tracker = is_left ? m_native_left_hand_tracker : m_native_right_hand_tracker;
    if (tracker == XR_NULL_HANDLE || !m_pfn_locate_hand_joints || time == 0)
    {
        return false;
    }

    XrHandJointsLocateInfoEXT locate_info{ XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT };
    locate_info.baseSpace = m_handles.space;
    locate_info.time = time;

    XrHandJointLocationEXT joint_locations[XR_HAND_JOINT_COUNT_EXT];

    XrHandJointLocationsEXT locations{ XR_TYPE_HAND_JOINT_LOCATIONS_EXT };
    locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
    locations.jointLocations = joint_locations;

    XrResult result = m_pfn_locate_hand_joints(tracker, &locate_info, &locations);
    if (XR_FAILED(result) || !locations.isActive)
    {
        return false;
    }

    const auto& wrist = joint_locations[XR_HAND_JOINT_WRIST_EXT];
    const bool is_valid = (wrist.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                          (wrist.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);
    if (!is_valid)
    {
        return false;
    }

    out_pose = wrist.pose;
    // Distinguish actively tracked from valid-but-predicted/stale poses so
    // callers can advertise TRACKED bits only when the runtime confirms it.
    out_tracked = (wrist.locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT) &&
                  (wrist.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT);
    return true;
}

bool WristPoseSource::query_controller(bool is_left, XrPosef& out_pose, bool& out_tracked)
{
    out_tracked = false;

    if (m_controller_tracker == nullptr || m_deviceio_session == nullptr)
    {
        return false;
    }

    const auto& tracked = is_left ? m_controller_tracker->get_left_controller(*m_deviceio_session) :
                                    m_controller_tracker->get_right_controller(*m_deviceio_session);
    if (!tracked)
    {
        return false;
    }

    bool aim_valid = false;
    const XrPosef aim_pose = oxr_utils::get_aim_pose(*tracked, aim_valid);
    if (!aim_valid)
    {
        return false;
    }

    const XrPosef& offset = is_left ? m_config.left_aim_to_wrist : m_config.right_aim_to_wrist;
    out_pose = oxr_utils::multiply_poses(aim_pose, offset);
    out_tracked = true;
    return true;
}

} // namespace plugin_utils
