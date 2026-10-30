// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/plugin_session.hpp>

#include <memory>
#include <string>
#include <vector>

namespace core
{
class OpenXRSession;
}

namespace plugin_utils
{

/*!
 * @brief Local plugin session that creates OpenXR-backed operation channels.
 *
 * The adapter owns the OpenXR session and its pull channel. OpenXR permits
 * action sets to be attached only once, so the pull channel is created with the
 * session. Its owner must destroy every returned push channel and wrist source
 * before destroying this session.
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

    core::IPluginPullChannel& pull_channel() override;
    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override;
    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT hand) override;

private:
    core::PluginSessionRequirements requirements_;
    std::shared_ptr<core::OpenXRSession> session_;
    std::unique_ptr<core::IPluginPullChannel> pull_channel_;
};

} // namespace plugin_utils
