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

class OpenXRWristTrackingSource final : public core::IWristTrackingSource
{
public:
    OpenXRWristTrackingSource(const core::WristTrackingSourceConfig& config,
                              const core::OpenXRSessionHandles& handles,
                              core::DeviceIOSession* deviceio_session,
                              std::shared_ptr<core::ControllerTracker> controller_tracker)
        : time_converter_(handles), source_(config, handles, deviceio_session, std::move(controller_tracker))
    {
    }

    core::WristTrackingSample query(XrHandEXT hand, int64_t sample_time_local_common_clock_ns) override
    {
        return source_.query(hand == XR_HAND_LEFT_EXT,
                             time_converter_.convert_monotonic_ns_to_xrtime(sample_time_local_common_clock_ns));
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
                            std::shared_ptr<core::ControllerTracker> wrist_controller_tracker)
        : handles_(handles),
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
        // The controller tracker exists exactly when wrist tracking pull was declared.
        if (!wrist_controller_tracker_)
        {
            throw std::logic_error(
                "OpenXRPluginSession: wrist tracking pull was not declared in PluginSessionRequirements");
        }
        return std::make_unique<OpenXRWristTrackingSource>(
            config, handles_, deviceio_session_.get(), wrist_controller_tracker_);
    }

private:
    core::OpenXRSessionHandles handles_;
    std::shared_ptr<core::ControllerTracker> wrist_controller_tracker_;
    std::unique_ptr<core::DeviceIOSession> deviceio_session_;
};

} // anonymous namespace

OpenXRPluginSession::OpenXRPluginSession(std::string app_name,
                                         core::PluginSessionRequirements requirements,
                                         std::vector<std::shared_ptr<core::ITracker>> trackers)
    : requirements_(requirements), trackers_(std::move(trackers))
{
    std::vector<std::string> wrist_extensions;
    if (requirements_.wrist_tracking_pull)
    {
        auto wrist = WristPoseSource::collect_requirements(WristSourceMode::Auto);
        wrist_controller_tracker_ = std::move(wrist.controller_tracker);
        trackers_.insert(trackers_.end(), wrist.trackers.begin(), wrist.trackers.end());
        wrist_extensions.assign(wrist.extensions.begin(), wrist.extensions.end());
    }

    auto extensions = make_required_extensions(requirements_, trackers_);
    append_unique(extensions, wrist_extensions);
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

    return std::make_unique<OpenXRPluginPullChannel>(session_->get_handles(), trackers_, wrist_controller_tracker_);
}

std::unique_ptr<core::ISchemaPushChannel> OpenXRPluginSession::create_schema_push_channel(
    const core::SchemaPusherConfig& config)
{
    if (!requirements_.schema_push)
    {
        throw std::logic_error("OpenXRPluginSession: schema push was not declared in PluginSessionRequirements");
    }
    return std::make_unique<core::OpenXRSchemaPushChannel>(session_->get_handles(), config);
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
