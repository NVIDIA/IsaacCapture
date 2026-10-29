// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/plugin_utils/openxr_plugin_session.hpp"

#include "inc/plugin_utils/wrist_pose_source.hpp"
#include "openxr_hand_tracking_push_channel.hpp"

#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/controller_tracker.hpp>
#include <openxr/openxr.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <pusherio/openxr_schema_push_channel.hpp>

#include <XR_MNDX_xdev_space.h>
#include <XR_NVX1_device_interface.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plugin_utils
{

namespace
{

void append_unique(std::vector<std::string>& extensions, const std::vector<std::string>& additions)
{
    for (const auto& extension : additions)
    {
        if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end())
        {
            extensions.push_back(extension);
        }
    }
}

std::vector<std::string> make_required_extensions(const core::PluginSessionRequirements& requirements,
                                                  const std::vector<std::shared_ptr<core::ITracker>>& trackers)
{
    auto extensions = core::DeviceIOSession::get_required_extensions(trackers);

    if (requirements.schema_push)
    {
        append_unique(extensions, core::OpenXRSchemaPushChannel::get_required_extensions());
    }

    if (requirements.hand_tracking_push)
    {
        append_unique(extensions, { XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME });
        append_unique(extensions, core::XrTimeConverter::get_required_extensions());
    }

    return extensions;
}

bool is_extension_supported(const char* extension_name)
{
    uint32_t count = 0;
    if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr)))
    {
        return false;
    }

    std::vector<XrExtensionProperties> properties(count, XrExtensionProperties{ XR_TYPE_EXTENSION_PROPERTIES });
    if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, count, &count, properties.data())))
    {
        return false;
    }

    return std::any_of(properties.begin(), properties.end(),
                       [extension_name](const XrExtensionProperties& property)
                       { return std::string(property.extensionName) == extension_name; });
}

WristSourceMode to_openxr_mode(core::WristTrackingSourceMode mode)
{
    switch (mode)
    {
    case core::WristTrackingSourceMode::Auto:
        return WristSourceMode::Auto;
    case core::WristTrackingSourceMode::HandTracking:
        return WristSourceMode::HandTracking;
    case core::WristTrackingSourceMode::Controller:
        return WristSourceMode::Controller;
    }
    throw std::logic_error("Unknown wrist tracking source mode");
}

WristSourceConfig to_openxr_config(const core::WristTrackingSourceConfig& config)
{
    return WristSourceConfig{ .mode = to_openxr_mode(config.mode),
                              .left_aim_to_wrist = config.left_aim_to_wrist,
                              .right_aim_to_wrist = config.right_aim_to_wrist };
}

class OpenXRWristTrackingSource final : public core::IWristTrackingSource
{
public:
    OpenXRWristTrackingSource(const core::WristTrackingSourceConfig& config,
                              const core::OpenXRSessionHandles& handles,
                              core::DeviceIOSession* deviceio_session,
                              std::shared_ptr<core::ControllerTracker> controller_tracker)
        : time_converter_(handles),
          source_(to_openxr_config(config), handles, deviceio_session, std::move(controller_tracker))
    {
    }

    core::WristTrackingSample query(bool is_left, int64_t sample_time_local_common_clock_ns) override
    {
        const WristSample sample =
            source_.query(is_left, time_converter_.convert_monotonic_ns_to_xrtime(sample_time_local_common_clock_ns));
        return core::WristTrackingSample{ .pose = sample.pose, .valid = sample.valid, .tracked = sample.tracked };
    }

private:
    core::XrTimeConverter time_converter_;
    WristPoseSource source_;
};

class OpenXRPluginPullChannel final : public core::IPluginPullChannel
{
public:
    OpenXRPluginPullChannel(const core::OpenXRSessionHandles& handles,
                            const std::vector<std::shared_ptr<core::ITracker>>& trackers,
                            bool wrist_tracking_enabled,
                            bool native_hand_tracking_enabled,
                            std::shared_ptr<core::ControllerTracker> wrist_controller_tracker)
        : handles_(handles),
          wrist_tracking_enabled_(wrist_tracking_enabled),
          native_hand_tracking_enabled_(native_hand_tracking_enabled),
          wrist_controller_tracker_(std::move(wrist_controller_tracker)),
          deviceio_session_(core::DeviceIOSession::run(trackers, handles_))
    {
    }

    void update() override
    {
        deviceio_session_->update();
    }

    const core::ITrackerImpl& get_tracker_impl(const core::ITracker& tracker) const override
    {
        return deviceio_session_->get_tracker_impl(tracker);
    }

    std::unique_ptr<core::IWristTrackingSource> create_wrist_tracking_source(
        const core::WristTrackingSourceConfig& config) override
    {
        if (!wrist_tracking_enabled_)
        {
            throw std::logic_error(
                "OpenXRPluginSession: wrist tracking pull was not declared in PluginSessionRequirements");
        }

        core::WristTrackingSourceConfig effective_config = config;
        if (!native_hand_tracking_enabled_)
        {
            if (effective_config.mode == core::WristTrackingSourceMode::HandTracking)
            {
                return nullptr;
            }
            if (effective_config.mode == core::WristTrackingSourceMode::Auto)
            {
                effective_config.mode = core::WristTrackingSourceMode::Controller;
            }
        }

        if (effective_config.mode != core::WristTrackingSourceMode::HandTracking && wrist_controller_tracker_ == nullptr)
        {
            return nullptr;
        }

        return std::make_unique<OpenXRWristTrackingSource>(
            effective_config, handles_, deviceio_session_.get(), wrist_controller_tracker_);
    }

private:
    core::OpenXRSessionHandles handles_;
    bool wrist_tracking_enabled_;
    bool native_hand_tracking_enabled_;
    std::shared_ptr<core::ControllerTracker> wrist_controller_tracker_;
    std::unique_ptr<core::DeviceIOSession> deviceio_session_;
};

} // anonymous namespace

OpenXRPluginSession::OpenXRPluginSession(std::string app_name,
                                         core::PluginSessionRequirements requirements,
                                         std::vector<std::shared_ptr<core::ITracker>> trackers)
    : requirements_(requirements), trackers_(std::move(trackers))
{
    if (requirements_.wrist_tracking_pull)
    {
        wrist_controller_tracker_ = std::make_shared<core::ControllerTracker>();
        trackers_.push_back(wrist_controller_tracker_);
        native_hand_tracking_enabled_ = is_extension_supported(XR_EXT_HAND_TRACKING_EXTENSION_NAME) &&
                                        is_extension_supported(XR_MNDX_XDEV_SPACE_EXTENSION_NAME);
    }

    auto extensions = make_required_extensions(requirements_, trackers_);
    if (native_hand_tracking_enabled_)
    {
        append_unique(extensions, { XR_EXT_HAND_TRACKING_EXTENSION_NAME, XR_MNDX_XDEV_SPACE_EXTENSION_NAME });
    }
    session_ = std::make_shared<core::OpenXRSession>(app_name, extensions);
}

OpenXRPluginSession::~OpenXRPluginSession() = default;

std::unique_ptr<core::IPluginPullChannel> OpenXRPluginSession::create_pull_channel()
{
    if (pull_channel_creation_attempted_)
    {
        throw std::logic_error("OpenXRPluginSession supports only one pull channel per session");
    }
    pull_channel_creation_attempted_ = true;

    return std::make_unique<OpenXRPluginPullChannel>(session_->get_handles(), trackers_,
                                                     requirements_.wrist_tracking_pull, native_hand_tracking_enabled_,
                                                     wrist_controller_tracker_);
}

std::unique_ptr<core::ISchemaPushChannel> OpenXRPluginSession::create_schema_push_channel(
    const core::SchemaPusherConfig& config)
{
    if (!requirements_.schema_push)
    {
        throw std::logic_error("OpenXRPluginSession: schema push was not declared in PluginSessionRequirements");
    }
    return core::make_openxr_schema_push_channel(session_->get_handles(), config);
}

std::unique_ptr<core::IHandTrackingPushChannel> OpenXRPluginSession::create_hand_tracking_push_channel(XrHandEXT hand)
{
    if (!requirements_.hand_tracking_push)
    {
        throw std::logic_error("OpenXRPluginSession: hand-tracking push was not declared in PluginSessionRequirements");
    }
    return make_openxr_hand_tracking_push_channel(session_->get_handles(), hand);
}

} // namespace plugin_utils
