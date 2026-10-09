// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <pusherio/schema_pusher.hpp>
#include <spacemouse/spacemouse_connection.hpp>

#include <memory>
#include <optional>
#include <string>

namespace core
{
class OpenXRSession;
}

namespace plugins
{
namespace spacemouse
{

struct SpaceMousePluginConfig
{
    SpaceMouseConnectionConfig connection;
    std::string collection_id = "spacemouse";
    //! Set by the plugin launcher; enables device status reporting for monitoring.
    std::optional<std::string> plugin_root_id;
};

/*!
 * @brief Reads a SpaceMouse and pushes SpaceMouseOutput via OpenXR SchemaPusher.
 *
 * A sample is pushed on every update, also while no device is open (connected = false, zero
 * axes), so the host never keeps the last deflection of a device that went away.
 */
class SpaceMousePlugin
{
public:
    //! Device path reported to plugin monitoring; matches plugin.yaml.
    static constexpr const char* DEVICE_PATH = "/spacemouse";

    explicit SpaceMousePlugin(SpaceMousePluginConfig config);
    ~SpaceMousePlugin();

    void update();

private:
    void push_state(int64_t now_ns);
    void publish_status(int64_t now_ns);

    SpaceMouseConnection connection_;
    std::shared_ptr<core::OpenXRSession> session_;
    core::SchemaPusher pusher_;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> status_;
};

} // namespace spacemouse
} // namespace plugins
