// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <avatar_sdk/AvatarSDK.h>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <openxr/openxr_platform.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <plugin_utils/wrist_pose_source.hpp>
#include <pusherio/schema_pusher.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace plugins
{
namespace avatar
{

using DeviceSide = ::avatar::DeviceSide;
using DeviceDataCategory = ::avatar::DeviceDataCategory;

inline constexpr std::array<DeviceSide, 2> kDeviceSides{ DeviceSide::LEFT, DeviceSide::RIGHT };
inline constexpr std::array<DeviceDataCategory, 2> kJointDataCategories{ DeviceDataCategory::RAW,
                                                                         DeviceDataCategory::ROBOT };

inline constexpr std::string_view to_string(DeviceSide side)
{
    switch (side)
    {
    case DeviceSide::LEFT:
        return "left";
    case DeviceSide::RIGHT:
        return "right";
    }
    return "unknown";
}

inline constexpr std::string_view to_string(DeviceDataCategory category)
{
    switch (category)
    {
    case DeviceDataCategory::RAW:
        return "RAW";
    case DeviceDataCategory::ROBOT:
        return "ROBOT";
    case DeviceDataCategory::HUMAN:
        return "HUMAN";
    }
    return "unknown";
}

inline constexpr uint32_t kAvatarHumanLandmarkCount = 25;

struct AvatarPluginConfig
{
    std::string app_name = "AvatarHandPlugin";
    std::string sdk_config_path;
    bool human = true;
    bool raw = true;
    bool robot = true;
    bool haptic = true;
};

struct AvatarPoint
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct AvatarQuaternion
{
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct AvatarLandmark
{
    AvatarPoint position;
    AvatarQuaternion orientation;
};

struct AvatarJointFrame
{
    std::vector<std::string> names;
    std::vector<float> positions;
};

class AvatarSdkSession
{
public:
    explicit AvatarSdkSession(const std::string& config_path);
    ~AvatarSdkSession() noexcept;

    AvatarSdkSession(const AvatarSdkSession&) = delete;
    AvatarSdkSession& operator=(const AvatarSdkSession&) = delete;

    ::avatar::AvatarSDK& get();
    const std::string& config_path() const;

private:
    std::string m_config_path;
};

struct GloveState
{
    ~GloveState() noexcept;

    GloveState() = default;
    GloveState(const GloveState&) = delete;
    GloveState& operator=(const GloveState&) = delete;

    void reset() noexcept;

    ::avatar::DevicePtr device;
    std::vector<::avatar::Pose> landmarks;
    ::avatar::AvatarDataFrame raw_frame;
    ::avatar::AvatarDataFrame robot_frame;
    // Empty until start() or a successful fetch; drives the data-timeout window.
    std::optional<std::chrono::steady_clock::time_point> last_successful_fetch;
};

class __attribute__((visibility("default"))) AvatarTracker
{
public:
    explicit AvatarTracker(AvatarPluginConfig config = {});

    void update();

    std::vector<AvatarLandmark> get_landmarks(DeviceSide side) const;
    AvatarJointFrame get_joint_frame(DeviceSide side, DeviceDataCategory category) const;

private:
    static constexpr std::size_t kAvatarFingerCount = 5;

    void initialize_openxr();
    void try_connect_missing_gloves();
    void start_glove_if_present(GloveState& glove, ::avatar::DeviceSide side);
    void refresh_data();
    void inject_hand_data();
    void push_joint_frames();
    void push_joint_frame(const GloveState& glove,
                          ::avatar::DeviceSide side,
                          ::avatar::DeviceDataCategory category,
                          core::SchemaPusher& pusher);
    bool dataset_enabled(::avatar::DeviceDataCategory category) const;
    void apply_haptic_command(::avatar::DeviceSide side, const std::array<float, kAvatarFingerCount>& powers);
    const GloveState& glove(::avatar::DeviceSide side) const;
    GloveState& glove(::avatar::DeviceSide side);
    void map_landmarks_to_openxr(const std::vector<::avatar::Pose>& landmarks,
                                 const XrPosef& root_pose,
                                 bool is_root_tracked,
                                 XrHandJointLocationEXT out_joints[XR_HAND_JOINT_COUNT_EXT]) const;

    AvatarPluginConfig m_config;
    AvatarSdkSession m_sdk;
    std::array<GloveState, 2> m_gloves;

    std::shared_ptr<core::OpenXRSession> m_session;
    core::OpenXRSessionHandles m_handles;
    std::array<std::unique_ptr<plugin_utils::HandInjector>, 2> m_injectors;
    std::shared_ptr<core::HapticCommandReaderTracker> m_haptic_reader;
    std::unique_ptr<core::DeviceIOSession> m_deviceio_session;
    std::optional<core::XrTimeConverter> m_time_converter;
    std::array<std::array<std::unique_ptr<core::SchemaPusher>, 2>, 2> m_joint_pushers;
    std::unique_ptr<plugin_utils::WristPoseSource> m_wrist_source;
    std::unordered_map<std::string, std::size_t> m_landmark_index;

    std::array<bool, 2> m_haptic_error_logged{ { false, false } };
    std::optional<std::chrono::steady_clock::time_point> m_last_glove_retry;
    std::optional<std::chrono::steady_clock::time_point> m_last_glove_wait_log;
};

} // namespace avatar
} // namespace plugins
