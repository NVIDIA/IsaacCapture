// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gamepad/gamepad_connection.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <pusherio/schema_pusher.hpp>

#include <memory>
#include <optional>
#include <string>

namespace core
{
class OpenXRSession;
}

namespace plugins
{
namespace gamepad
{

struct GamepadPluginConfig
{
    GamepadConnectionConfig connection;
    std::string collection_id = "gamepad";
    //! Set by the plugin launcher; enables device status reporting for monitoring.
    std::optional<std::string> plugin_root_id;
};

/*!
 * @brief Reads a gamepad and pushes GamepadSnapshot via OpenXR SchemaPusher.
 *
 * A sample is pushed on every update, also while no device is open (no inputs, nothing
 * held), so the host never keeps the last stick deflection of a pad that went away.
 */
class GamepadPlugin
{
public:
    //! Device path reported to plugin monitoring; matches plugin.yaml.
    static constexpr const char* DEVICE_PATH = "/gamepad";

    explicit GamepadPlugin(GamepadPluginConfig config);
    ~GamepadPlugin();

    void update();

private:
    void push_state(int64_t now_ns);
    void publish_status(int64_t now_ns);

    GamepadConnection connection_;
    std::shared_ptr<core::OpenXRSession> session_;
    core::SchemaPusher pusher_;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> status_;
};

} // namespace gamepad
} // namespace plugins
