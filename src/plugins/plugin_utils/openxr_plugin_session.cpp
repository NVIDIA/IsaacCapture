// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/plugin_utils/openxr_plugin_session.hpp"

#include "inc/plugin_utils/hand_injector.hpp"
#include "inc/plugin_utils/wrist_pose_source.hpp"
#include "openxr_hand_tracking_push_channel.hpp"

#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/controller_tracker.hpp>
#include <openxr/openxr.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <pusherio/openxr_schema_push_channel.hpp>

#include <algorithm>
#include <optional>
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
                                                  const std::vector<std::shared_ptr<core::ITracker>>& trackers,
                                                  const std::vector<const char*>& wrist_extensions)
{
    auto extensions = core::DeviceIOSession::get_required_extensions(trackers);
    append_unique(extensions, std::vector<std::string>(wrist_extensions.begin(), wrist_extensions.end()));

    if (requirements.schema_push)
    {
        append_unique(extensions, core::OpenXRSchemaPushChannel::get_required_extensions());
    }

    if (requirements.hand_tracking_push)
    {
        append_unique(extensions, HandInjector::get_required_extensions());
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
                            const std::optional<core::WristTrackingSourceConfig>& wrist_config,
                            std::shared_ptr<core::ControllerTracker> wrist_controller_tracker)
        : deviceio_session_(core::DeviceIOSession::run(trackers, handles))
    {
        if (wrist_config)
        {
            wrist_source_ = std::make_unique<OpenXRWristTrackingSource>(
                *wrist_config, handles, deviceio_session_.get(), std::move(wrist_controller_tracker));
        }
    }

    void update() override
    {
        deviceio_session_->update();
    }

    const core::ITrackerImpl& get_tracker_impl(const core::ITracker& tracker) const override
    {
        return deviceio_session_->get_tracker_impl(tracker);
    }

    core::IWristTrackingSource& wrist_tracking_source() override
    {
        if (!wrist_source_)
        {
            throw std::logic_error(
                "OpenXRPluginSession: wrist tracking pull was not declared in PluginSessionRequirements");
        }
        return *wrist_source_;
    }

private:
    std::unique_ptr<core::DeviceIOSession> deviceio_session_;
    std::unique_ptr<OpenXRWristTrackingSource> wrist_source_;
};

class OpenXRPluginSession final : public core::IPluginSession
{
public:
    OpenXRPluginSession(const std::string& app_name,
                        const core::PluginSessionRequirements& requirements,
                        std::vector<std::shared_ptr<core::ITracker>> trackers)
        : requirements_(requirements)
    {
        WristPoseSource::Requirements wrist;
        if (requirements_.wrist_tracking_pull)
        {
            wrist = WristPoseSource::collect_requirements(requirements_.wrist_tracking_pull->mode);
            trackers.insert(trackers.end(), wrist.trackers.begin(), wrist.trackers.end());
        }

        session_ = std::make_unique<core::OpenXRSession>(
            app_name, make_required_extensions(requirements_, trackers, wrist.extensions));
        pull_channel_ = std::make_unique<OpenXRPluginPullChannel>(
            session_->get_handles(), trackers, requirements_.wrist_tracking_pull, std::move(wrist.controller_tracker));
    }

    core::IPluginPullChannel& pull_channel() override
    {
        return *pull_channel_;
    }

    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override
    {
        if (!requirements_.schema_push)
        {
            throw std::logic_error("OpenXRPluginSession: schema push was not declared in PluginSessionRequirements");
        }
        return std::make_unique<core::OpenXRSchemaPushChannel>(session_->get_handles(), config);
    }

    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT hand) override
    {
        if (!requirements_.hand_tracking_push)
        {
            throw std::logic_error(
                "OpenXRPluginSession: hand-tracking push was not declared in PluginSessionRequirements");
        }
        return std::make_unique<OpenXRHandTrackingPushChannel>(session_->get_handles(), hand);
    }

private:
    core::PluginSessionRequirements requirements_;
    std::unique_ptr<core::OpenXRSession> session_;
    std::unique_ptr<core::IPluginPullChannel> pull_channel_;
};

} // anonymous namespace

core::PluginSessionFactory make_openxr_plugin_session_factory(std::string app_name)
{
    return [app_name = std::move(app_name)](const core::PluginSessionRequirements& requirements,
                                            std::vector<std::shared_ptr<core::ITracker>> trackers)
    { return std::make_unique<OpenXRPluginSession>(app_name, requirements, std::move(trackers)); };
}

} // namespace plugin_utils
