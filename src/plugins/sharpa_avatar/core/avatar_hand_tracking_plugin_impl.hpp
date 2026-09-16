// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "avatar_glove_collection.hpp"
#include "inc/avatar/avatar_hand_tracking_plugin.hpp"

#include <avatar_sdk/AvatarSDK.h>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/controller_tracker.hpp>
#include <deviceio_trackers/hand_tracker.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <openxr/openxr_platform.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <pusherio/schema_pusher.hpp>

#include <XR_MNDX_xdev_space.h>
#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace plugins
{
namespace avatar
{

class AvatarSdkSession
{
public:
    explicit AvatarSdkSession(const std::string& config_path);
    ~AvatarSdkSession();

    AvatarSdkSession(const AvatarSdkSession&) = delete;
    AvatarSdkSession& operator=(const AvatarSdkSession&) = delete;

    ::avatar::AvatarSDK& get();

private:
    bool m_initialized = false;
};

struct GloveState
{
    ~GloveState();

    GloveState() = default;
    GloveState(const GloveState&) = delete;
    GloveState& operator=(const GloveState&) = delete;

    void reset();

    ::avatar::DevicePtr device;
    std::vector<::avatar::Pose> landmarks;
    ::avatar::AvatarDataFrame raw_frame;
    ::avatar::AvatarDataFrame robot_frame;
    bool started = false;
    std::chrono::steady_clock::time_point last_successful_fetch{};
};

class __attribute__((visibility("hidden"))) AvatarTracker::Impl
{
public:
    explicit Impl(AvatarPluginConfig config);
    ~Impl();

    void update();

    std::vector<AvatarLandmark> get_left_landmarks() const;
    std::vector<AvatarLandmark> get_right_landmarks() const;
    AvatarJointFrame get_left_raw_frame() const;
    AvatarJointFrame get_right_raw_frame() const;
    AvatarJointFrame get_left_robot_frame() const;
    AvatarJointFrame get_right_robot_frame() const;

private:
    void try_initialize_openxr();
    void reset_openxr();
    void connect_gloves();
    void try_connect_missing_gloves();
    void start_glove_if_present(GloveState& glove, const char* label);
    void refresh_data();
    void inject_hand_data();
    void push_joint_frames();
    void push_joint_frame(bool is_left, bool is_robot, core::SchemaPusher& pusher);
    void apply_haptic_command(bool is_left, const std::vector<float>& powers);
    void initialize_xdev_hand_trackers();
    void cleanup_xdev_hand_trackers();
    bool update_xdev_hand(XrHandTrackerEXT tracker, XrTime time, XrPosef& out_wrist_pose, bool& out_is_tracked);
    bool get_controller_wrist_pose(bool is_left, XrPosef& out_wrist_pose);
    void map_landmarks_to_openxr(const std::vector<::avatar::Pose>& landmarks,
                                 const XrPosef& root_pose,
                                 bool is_root_tracked,
                                 XrHandJointLocationEXT out_joints[XR_HAND_JOINT_COUNT_EXT]) const;

    AvatarPluginConfig m_config;
    AvatarSdkSession m_sdk;
    GloveState m_left;
    GloveState m_right;

    std::shared_ptr<core::OpenXRSession> m_session;
    core::OpenXRSessionHandles m_handles;
    std::unique_ptr<plugin_utils::HandInjector> m_left_injector;
    std::unique_ptr<plugin_utils::HandInjector> m_right_injector;
    std::shared_ptr<core::ControllerTracker> m_controller_tracker;
    std::shared_ptr<core::HandTracker> m_hand_tracker;
    std::shared_ptr<core::HapticCommandReaderTracker> m_haptic_reader;
    std::unique_ptr<core::DeviceIOSession> m_deviceio_session;
    std::optional<core::XrTimeConverter> m_time_converter;
    std::unique_ptr<core::SchemaPusher> m_left_raw_pusher;
    std::unique_ptr<core::SchemaPusher> m_right_raw_pusher;
    std::unique_ptr<core::SchemaPusher> m_left_robot_pusher;
    std::unique_ptr<core::SchemaPusher> m_right_robot_pusher;

    XrPosef m_left_root_pose = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
    XrPosef m_right_root_pose = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };

    bool m_xdev_available = false;
    XrXDevListMNDX m_xdev_list = XR_NULL_HANDLE;
    XrHandTrackerEXT m_native_left_hand_tracker = XR_NULL_HANDLE;
    XrHandTrackerEXT m_native_right_hand_tracker = XR_NULL_HANDLE;
    PFN_xrCreateXDevListMNDX m_pfn_create_xdev_list = nullptr;
    PFN_xrDestroyXDevListMNDX m_pfn_destroy_xdev_list = nullptr;
    PFN_xrEnumerateXDevsMNDX m_pfn_enumerate_xdevs = nullptr;
    PFN_xrGetXDevPropertiesMNDX m_pfn_get_xdev_properties = nullptr;
    PFN_xrCreateHandTrackerEXT m_pfn_create_hand_tracker = nullptr;
    PFN_xrDestroyHandTrackerEXT m_pfn_destroy_hand_tracker = nullptr;
    PFN_xrLocateHandJointsEXT m_pfn_locate_hand_joints = nullptr;

    std::array<bool, 2> m_haptic_error_logged{ { false, false } };
    std::chrono::steady_clock::time_point m_last_glove_retry{};
    std::chrono::steady_clock::time_point m_last_glove_wait_log{};
    std::chrono::steady_clock::time_point m_last_openxr_retry{};
    std::mutex m_update_mutex;
    mutable std::mutex m_data_mutex;
};

} // namespace avatar
} // namespace plugins
