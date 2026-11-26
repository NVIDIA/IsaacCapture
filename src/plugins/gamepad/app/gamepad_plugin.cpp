// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gamepad_plugin.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/gamepad_generated.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plugins
{
namespace gamepad
{

namespace
{

// Matches max_flatbuffer_size in trackers.toml (the default), which the host's GamepadTracker reads
// with; a sample is under 100 bytes.
constexpr size_t kMaxFlatbufferSize = 512;

std::vector<std::string> required_extensions(bool monitoring)
{
    std::vector<std::string> extensions = core::SchemaPusher::get_required_extensions();
    if (monitoring)
    {
        for (const auto& extension : plugin_utils::PluginDeviceStatusPublisher::get_required_extensions())
        {
            if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end())
                extensions.push_back(extension);
        }
    }
    return extensions;
}

std::shared_ptr<core::OpenXRSession> make_session(const GamepadPluginConfig& config)
{
    if (config.plugin_root_id && config.plugin_root_id->empty())
        throw std::invalid_argument("GamepadPlugin: --plugin-root-id must not be empty");
    return std::make_shared<core::OpenXRSession>("GamepadPlugin", required_extensions(config.plugin_root_id.has_value()));
}

} // namespace

GamepadPlugin::GamepadPlugin(GamepadPluginConfig config)
    : connection_(std::move(config.connection)),
      session_(make_session(config)),
      pusher_(session_->get_handles(),
              core::SchemaPusherConfig{ .collection_id = config.collection_id,
                                        .max_flatbuffer_size = kMaxFlatbufferSize,
                                        .tensor_identifier = "gamepad",
                                        .localized_name = "Gamepad",
                                        .app_name = "GamepadPlugin" })
{
    if (config.plugin_root_id)
        status_ =
            std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(session_->get_handles(), *config.plugin_root_id);
}

GamepadPlugin::~GamepadPlugin() = default;

void GamepadPlugin::update()
{
    const int64_t now_ns = core::os_monotonic_now_ns();
    connection_.poll();
    push_state(now_ns);
    publish_status(now_ns);
}

void GamepadPlugin::push_state(int64_t now_ns)
{
    core::GamepadSnapshotT out;
    // No inputs while no pad is open, so readers drop the last input of a pad that went away.
    if (connection_.status() == ConnectionStatus::Connected)
        out.inputs = std::make_shared<core::GamepadInputState>(connection_.state().inputs);

    flatbuffers::FlatBufferBuilder builder(kMaxFlatbufferSize);
    builder.Finish(core::GamepadSnapshot::Pack(builder, &out));
    pusher_.push_buffer(builder.GetBufferPointer(), builder.GetSize(), now_ns, now_ns);
}

void GamepadPlugin::publish_status(int64_t now_ns)
{
    if (!status_)
        return;

    plugin_utils::PluginDeviceStatusEntry entry{ .path = DEVICE_PATH };
    switch (connection_.status())
    {
    case ConnectionStatus::Connected:
        entry.state = core::PluginDeviceState_CONNECTED;
        entry.reason = core::PluginDeviceReason_NONE;
        break;
    case ConnectionStatus::NoDevice:
        entry.state = core::PluginDeviceState_DISCONNECTED;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        break;
    case ConnectionStatus::OpenFailed:
        entry.state = core::PluginDeviceState_FAILED;
        entry.reason = core::PluginDeviceReason_DEVICE_ERROR;
        entry.error = connection_.error();
        break;
    }
    status_->publish_if_changed({ entry }, now_ns);
}

} // namespace gamepad
} // namespace plugins
