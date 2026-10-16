// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spacemouse_plugin.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/spacemouse_generated.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plugins
{
namespace spacemouse
{

namespace
{

// Matches max_flatbuffer_size in trackers.toml, which the host's SpaceMouseTracker reads with.
constexpr size_t kMaxFlatbufferSize = 256;

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

std::shared_ptr<core::OpenXRSession> make_session(const SpaceMousePluginConfig& config)
{
    if (config.plugin_root_id && config.plugin_root_id->empty())
        throw std::invalid_argument("SpaceMousePlugin: --plugin-root-id must not be empty");
    return std::make_shared<core::OpenXRSession>(
        "SpaceMousePlugin", required_extensions(config.plugin_root_id.has_value()));
}

} // namespace

SpaceMousePlugin::SpaceMousePlugin(SpaceMousePluginConfig config)
    : connection_(std::move(config.connection)),
      session_(make_session(config)),
      pusher_(session_->get_handles(),
              core::SchemaPusherConfig{ .collection_id = config.collection_id,
                                        .max_flatbuffer_size = kMaxFlatbufferSize,
                                        .tensor_identifier = "spacemouse",
                                        .localized_name = "SpaceMouse",
                                        .app_name = "SpaceMousePlugin" })
{
    if (config.plugin_root_id)
        status_ =
            std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(session_->get_handles(), *config.plugin_root_id);
}

SpaceMousePlugin::~SpaceMousePlugin() = default;

void SpaceMousePlugin::update()
{
    const int64_t now_ns = core::os_monotonic_now_ns();
    connection_.poll(now_ns);
    push_state(now_ns);
    publish_status(now_ns);
}

void SpaceMousePlugin::push_state(int64_t now_ns)
{
    const SpaceMouseState& state = connection_.state();
    core::SpaceMouseOutputT out;
    out.translation = std::make_shared<core::Point>(state.translation[0], state.translation[1], state.translation[2]);
    out.rotation = std::make_shared<core::Point>(state.rotation[0], state.rotation[1], state.rotation[2]);
    out.buttons = state.buttons;
    out.connected = connection_.status() == ConnectionStatus::Connected;

    flatbuffers::FlatBufferBuilder builder(kMaxFlatbufferSize);
    builder.Finish(core::SpaceMouseOutput::Pack(builder, &out));
    pusher_.push_buffer(builder.GetBufferPointer(), builder.GetSize(), now_ns, now_ns);
}

void SpaceMousePlugin::publish_status(int64_t now_ns)
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

} // namespace spacemouse
} // namespace plugins
