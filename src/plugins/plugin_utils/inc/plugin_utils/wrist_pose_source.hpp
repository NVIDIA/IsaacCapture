// SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device wrist-pose source for glove plugins: optical (HMD) hand tracking via
// XR_MNDX_xdev_space preferred, controller aim pose as the fallback.
#pragma once

#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/controller_tracker.hpp>
#include <openxr/openxr.h>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/pose_conversions.hpp>

#include <XR_MNDX_xdev_space.h>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace plugin_utils
{

/** @brief Which hand a wrist pose belongs to. */
enum class WristSide
{
    Left = 0,
    Right = 1,
};

//! Number of hands a wrist source serves; WristSide order matches array indices.
inline constexpr size_t kSideCount = 2;

/** @brief Which device provides the wrist pose. */
enum class WristSourceMode
{
    Auto, //!< Optical hand tracking preferred, controller aim pose as fallback.
    HandTracking, //!< XR_MNDX_xdev_space optical hand tracking only.
    Controller, //!< Controller aim pose (plus the configured rigid offset) only.
};

/** @brief Source selection plus per-device tuning. */
struct WristSourceConfig
{
    WristSourceMode mode = WristSourceMode::Auto;
    // Rigid transform from the controller aim pose to the wrist joint, per
    // hand. Depends on the controller model and how it is mounted on the
    // glove, so each plugin supplies its own calibrated pair (see
    // kLeft/kRightAimToWrist in wuji_glove_plugin.cpp and
    // kLeft/kRightHandOffset in manus_hand_tracking_plugin.cpp).
    // Indexed by WristSide so the two hands are a table lookup, not a branch.
    // Defaults to the identity pose: XrPosef is a C struct with no default member
    // initializers, so `XrPosef{}` would give an all-zero, non-unit quaternion.
    std::array<XrPosef, kSideCount> aim_to_wrist = { oxr_utils::identity_posef(), oxr_utils::identity_posef() };
};

/** @brief One wrist query result, in the session base space. */
struct WristSample
{
    XrPosef pose = oxr_utils::identity_posef();
    bool valid = false; //!< Pose usable (may be the cached last good pose).
    bool tracked = false; //!< Source actively tracked this frame.
};

/**
 * @brief Selects an optical or controller-derived wrist pose.
 *
 * Optical hand tracking is preferred; a controller aim pose plus a configured
 * rigid offset is the fallback. A valid-but-untracked optical pose is retained
 * to avoid source jumps. When neither source yields a pose, the last valid pose
 * is returned as untracked.
 */
class WristPoseSource
{
public:
    /**
     * @brief Session prerequisites for a mode.
     *
     * Append `extensions` to the OpenXR session's extension list and pass
     * `trackers` into the DeviceIOSession before creating either; then hand
     * `controller_tracker` (null unless the controller source is enabled) to
     * the constructor.
     */
    struct Requirements
    {
        std::vector<std::shared_ptr<core::ITracker>> trackers;
        std::vector<const char*> extensions;
        std::shared_ptr<core::ControllerTracker> controller_tracker;
    };
    static Requirements collect_requirements(WristSourceMode mode);

    /**
     * @param deviceio_session Non-owning; must outlive this object. May be
     *        null when the controller source is not used.
     */
    WristPoseSource(const WristSourceConfig& config,
                    const core::OpenXRSessionHandles& handles,
                    core::DeviceIOSession* deviceio_session,
                    std::shared_ptr<core::ControllerTracker> controller_tracker);
    ~WristPoseSource();

    WristPoseSource(const WristPoseSource&) = delete;
    WristPoseSource& operator=(const WristPoseSource&) = delete;
    WristPoseSource(WristPoseSource&&) = delete;
    WristPoseSource& operator=(WristPoseSource&&) = delete;

    /**
     * @brief Query one hand's wrist pose.
     *
     * Call from the thread that pumps the DeviceIOSession (the trackers'
     * queries are not synchronized with concurrent update() calls).
     */
    WristSample query(WristSide side, XrTime time);

private:
    void initialize_xdev_hand_trackers();
    void cleanup_xdev_hand_trackers();
    bool query_xdev(WristSide side, XrTime time, XrPosef& out_pose, bool& out_tracked);
    bool query_controller(WristSide side, XrPosef& out_pose, bool& out_tracked);

    WristSourceConfig m_config;
    core::OpenXRSessionHandles m_handles;
    core::DeviceIOSession* m_deviceio_session; // non-owning
    std::shared_ptr<core::ControllerTracker> m_controller_tracker;

    struct HandState
    {
        XrPosef last_pose = oxr_utils::identity_posef();
        bool has_pose = false;
    };
    std::array<HandState, kSideCount> m_hand_state;

    // Optical hand tracking via XR_MNDX_xdev_space.
    XrXDevListMNDX m_xdev_list = XR_NULL_HANDLE;
    std::array<XrHandTrackerEXT, kSideCount> m_native_hand_tracker{};
    bool m_xdev_available = false;

    PFN_xrCreateXDevListMNDX m_pfn_create_xdev_list = nullptr;
    PFN_xrDestroyXDevListMNDX m_pfn_destroy_xdev_list = nullptr;
    PFN_xrEnumerateXDevsMNDX m_pfn_enumerate_xdevs = nullptr;
    PFN_xrGetXDevPropertiesMNDX m_pfn_get_xdev_properties = nullptr;
    PFN_xrCreateHandTrackerEXT m_pfn_create_hand_tracker = nullptr;
    PFN_xrDestroyHandTrackerEXT m_pfn_destroy_hand_tracker = nullptr;
    PFN_xrLocateHandJointsEXT m_pfn_locate_hand_joints = nullptr;
};

} // namespace plugin_utils
