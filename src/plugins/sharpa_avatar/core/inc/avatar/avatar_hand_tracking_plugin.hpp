// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "avatar_health.hpp"

#include <avatar_sdk/AvatarSDK.h>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <openxr/openxr_platform.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <plugin_utils/wrist_pose_source.hpp>
#include <pusherio/schema_pusher.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace plugins
{
namespace avatar
{

using DeviceSide = ::avatar::DeviceSide;
using DeviceDataCategory = ::avatar::DeviceDataCategory;

inline constexpr std::array<DeviceSide, 2> kDeviceSides{ DeviceSide::LEFT, DeviceSide::RIGHT };
inline constexpr std::array<DeviceDataCategory, 3> kDataCategories{ DeviceDataCategory::HUMAN, DeviceDataCategory::RAW,
                                                                    DeviceDataCategory::ROBOT };
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
    std::optional<std::string> monitoring_plugin_root_id;
    std::string sdk_config_path;
    bool human = true;
    bool raw = true;
    bool robot = true;
    bool haptic = true;
};

class AvatarSdkSession
{
public:
    explicit AvatarSdkSession(const std::string& config_path);
    ~AvatarSdkSession() noexcept;

    AvatarSdkSession(const AvatarSdkSession&) = delete;
    AvatarSdkSession& operator=(const AvatarSdkSession&) = delete;

    ::avatar::AvatarSDK& get();
};

class GloveIngressSignal
{
public:
    void record() noexcept;
    uint64_t generation() const noexcept;
    std::optional<std::chrono::steady_clock::time_point> last_event() const noexcept;

private:
    std::atomic<uint64_t> m_generation{ 0 };
    std::atomic<int64_t> m_last_event_ns{ -1 };
};

struct GloveState
{
    ~GloveState() noexcept;

    GloveState() = default;
    GloveState(const GloveState&) = delete;
    GloveState& operator=(const GloveState&) = delete;

    void reset() noexcept;
    void clear_current_data() noexcept;

    struct DatasetState
    {
        std::optional<::avatar::Stamp> last_stamp;
        std::optional<std::chrono::steady_clock::time_point> last_fresh_data;
        std::optional<::avatar::ErrorCode> fetch_error;
        bool ever_received_data = false;
    };

    ::avatar::DevicePtr device;
    std::vector<::avatar::Pose> landmarks;
    ::avatar::AvatarDataFrame raw_frame;
    ::avatar::AvatarDataFrame robot_frame;
    std::array<DatasetState, kDataCategories.size()> datasets;
    std::shared_ptr<GloveIngressSignal> ingress;
    uint64_t observed_ingress_generation = 0;
    std::optional<std::chrono::steady_clock::time_point> last_ingress;
    std::optional<std::string> haptic_error;
    std::optional<std::string> output_setup_error;
    std::optional<std::chrono::steady_clock::time_point> last_output_setup_attempt;
    bool ever_received_ingress = false;
    bool started = false;
    bool ready = false;
};

class HapticCommandDeliveryState
{
public:
    using Clock = std::chrono::steady_clock;

    void observe(const core::Serialized<core::HapticCommand>& command, Clock::time_point observed_at)
    {
        if (command.get() != m_latest.get())
        {
            m_latest = command;
            m_pending_since = command.get() != nullptr ? std::make_optional(observed_at) : std::nullopt;
        }
    }

    const core::HapticCommand* pending(Clock::time_point now, Clock::duration freshness_timeout) noexcept
    {
        if (!m_pending_since || now - *m_pending_since >= freshness_timeout)
        {
            m_pending_since.reset();
            return nullptr;
        }
        return m_latest.get();
    }

    void acknowledge() noexcept
    {
        m_pending_since.reset();
    }

    void discard() noexcept
    {
        // Retain the identity so a cached tracker value cannot become a new command.
        m_pending_since.reset();
    }

private:
    core::Serialized<core::HapticCommand> m_latest;
    std::optional<Clock::time_point> m_pending_since;
};

class __attribute__((visibility("default"))) AvatarTracker
{
public:
    explicit AvatarTracker(AvatarPluginConfig config = {});
    ~AvatarTracker() noexcept;

    void update();

private:
    static constexpr std::size_t kAvatarFingerCount = 5;

    void initialize_openxr();
    void refresh_transport_state() noexcept;
    void refresh_sdk_inventory();
    void reconcile_glove_handles();
    void reset_glove(::avatar::DeviceSide side) noexcept;
    void try_connect_missing_gloves();
    void start_glove_if_present(GloveState& glove, ::avatar::DeviceSide side);
    void refresh_glove_activity();
    bool try_create_output_resources(::avatar::DeviceSide side);
    void destroy_output_resources(::avatar::DeviceSide side) noexcept;
    void refresh_data();
    void publish_device_status();
    void process_haptics();
    void inject_hand_data();
    void push_joint_frames();
    void push_joint_frame(const GloveState& glove,
                          ::avatar::DeviceSide side,
                          ::avatar::DeviceDataCategory category,
                          core::SchemaPusher& pusher);
    bool dataset_enabled(::avatar::DeviceDataCategory category) const;
    bool apply_haptic_command(::avatar::DeviceSide side, const std::array<float, kAvatarFingerCount>& powers);
    bool send_haptic_command(::avatar::DeviceSide side, const std::array<float, kAvatarFingerCount>& powers);
    void stop_haptics(::avatar::DeviceSide side) noexcept;
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
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> m_status_publisher;
    std::shared_ptr<core::HapticCommandReaderTracker> m_haptic_reader;
    std::unique_ptr<core::DeviceIOSession> m_deviceio_session;
    std::optional<core::XrTimeConverter> m_time_converter;
    std::array<std::array<std::unique_ptr<core::SchemaPusher>, 2>, 2> m_joint_pushers;
    std::unique_ptr<plugin_utils::WristPoseSource> m_wrist_source;

    std::array<HapticCommandDeliveryState, 2> m_haptic_commands;
    std::array<std::optional<std::chrono::steady_clock::time_point>, 2> m_last_haptic_sample_times;
    std::array<std::optional<std::chrono::steady_clock::time_point>, 2> m_haptic_retry_after;
    std::array<bool, 2> m_haptic_stopped{ { true, true } };
    std::array<bool, 2> m_haptic_error_logged{ { false, false } };
    std::optional<std::chrono::steady_clock::time_point> m_last_glove_retry;
    std::array<std::optional<::avatar::DeviceInfo>, 2> m_inventory;
    std::array<bool, 2> m_ever_observed{};
    bool m_inventory_available = false;
    std::optional<std::chrono::steady_clock::time_point> m_last_inventory_refresh;
    std::optional<std::chrono::steady_clock::time_point> m_last_dongle_check;
    UsbDevicePresence m_dongle_presence = UsbDevicePresence::UNKNOWN;
    ::avatar::UsbConnectionState m_transport_state = ::avatar::UsbConnectionState::DISCONNECTED;
    bool m_glove_wait_logged = false;
    bool m_inventory_error_logged = false;
    bool m_status_publish_error_logged = false;
};

} // namespace avatar
} // namespace plugins
