// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/plugin_session.hpp>

#include <memory>
#include <string>
#include <vector>

namespace core
{
class ControllerTracker;
class ITracker;
class OpenXRSession;
}

namespace plugin_utils
{

/*!
 * @brief Local plugin session that creates OpenXR-backed operation channels.
 *
 * The adapter owns the OpenXR session. Its owner must destroy every returned
 * pull and push channel before destroying this session. OpenXR permits action
 * sets to be attached only once, so create_pull_channel() may be called once.
 */
class OpenXRPluginSession final : public core::IPluginSession
{
public:
    OpenXRPluginSession(std::string app_name,
                        core::PluginSessionRequirements requirements,
                        std::vector<std::shared_ptr<core::ITracker>> trackers = {});
    ~OpenXRPluginSession() override;

    OpenXRPluginSession(const OpenXRPluginSession&) = delete;
    OpenXRPluginSession& operator=(const OpenXRPluginSession&) = delete;
    OpenXRPluginSession(OpenXRPluginSession&&) = delete;
    OpenXRPluginSession& operator=(OpenXRPluginSession&&) = delete;

    std::unique_ptr<core::IPluginPullChannel> create_pull_channel() override;
    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override;
    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT hand) override;

private:
    core::PluginSessionRequirements requirements_;
    std::vector<std::shared_ptr<core::ITracker>> trackers_;
    std::shared_ptr<core::ControllerTracker> wrist_controller_tracker_;
    bool native_hand_tracking_enabled_ = false;
    bool pull_channel_creation_attempted_ = false;
    std::shared_ptr<core::OpenXRSession> session_;
};

} // namespace plugin_utils
