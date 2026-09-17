// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_plugin.hpp"

#include "realhand_ffg_glove_serial.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <schema/joint_state_generated.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <thread>
#include <utility>

namespace plugins::realhand_ffg_glove
{

namespace
{

constexpr std::size_t kMaxFlatbufferSize = 4096;
constexpr std::size_t kForceChannelCount = 5;
constexpr auto kQueryPeriod = std::chrono::milliseconds(16);
constexpr auto kStaleTimeout = std::chrono::seconds(1);
constexpr auto kDiscoveryPeriod = std::chrono::seconds(2);
constexpr std::array<const char*, 2> kCollectionIds = { "realhand_ffg_glove_left", "realhand_ffg_glove_right" };
constexpr std::array<const char*, 2> kDevicePaths = { "/glove/realhand_ffg_glove_left",
                                                      "/glove/realhand_ffg_glove_right" };

std::size_t side_index(HandSide side)
{
    return side == HandSide::Left ? 0 : 1;
}

const char* side_name(std::size_t side)
{
    return side == 0 ? "left" : "right";
}

std::vector<std::string> required_extensions()
{
    auto extensions = core::SchemaPusher::get_required_extensions();
    for (const auto& extension : plugin_utils::PluginDeviceStatusPublisher::get_required_extensions())
    {
        if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end())
        {
            extensions.push_back(extension);
        }
    }
    return extensions;
}

} // namespace

RealHandFFGGlovePlugin::RealHandFFGGlovePlugin(std::vector<std::string> configured_ports,
                                               std::vector<int> baudrates,
                                               std::string plugin_root_id)
    : configured_ports_(std::move(configured_ports)),
      baudrates_(std::move(baudrates)),
      session_(std::make_shared<core::OpenXRSession>("RealHandFFGGlovePlugin", required_extensions()))
{
    if (baudrates_.empty())
    {
        baudrates_.assign(kDefaultBaudrates.begin(), kDefaultBaudrates.end());
    }
    for (std::size_t side = 0; side < pushers_.size(); ++side)
    {
        pushers_[side] = std::make_unique<core::SchemaPusher>(
            session_->get_handles(),
            core::SchemaPusherConfig{ .collection_id = kCollectionIds[side],
                                      .max_flatbuffer_size = kMaxFlatbufferSize,
                                      .tensor_identifier = "joint_state",
                                      .localized_name = std::string("RealHand FFG Glove ") + side_name(side),
                                      .app_name = "RealHandFFGGlovePlugin" });
    }
    status_publisher_ =
        std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(session_->get_handles(), plugin_root_id);
    last_discovery_ = std::chrono::steady_clock::time_point::min();
}

RealHandFFGGlovePlugin::~RealHandFFGGlovePlugin() = default;

void RealHandFFGGlovePlugin::discover()
{
    const auto now = std::chrono::steady_clock::now();
    if (now - last_discovery_ < kDiscoveryPeriod)
    {
        return;
    }
    last_discovery_ = now;

    std::vector<std::string> ports = configured_ports_.empty() ? discover_serial_ports() : configured_ports_;
    std::set<std::string> occupied;
    for (const auto& endpoint : endpoints_)
    {
        if (endpoint)
        {
            occupied.insert(endpoint->serial->port());
        }
    }

    for (const auto& port : ports)
    {
        if (occupied.contains(port))
        {
            continue;
        }
        for (int baudrate : baudrates_)
        {
            try
            {
                auto serial = std::make_unique<SerialGlove>(port, baudrate);
                auto version = probe_glove(*serial);
                if (!version)
                {
                    continue;
                }
                const std::size_t side = side_index(version->side);
                if (endpoints_[side])
                {
                    std::cerr << "RealHandFFGGlovePlugin: ignoring duplicate " << side_name(side) << " glove on "
                              << port << std::endl;
                    break;
                }
                auto endpoint = std::make_unique<Endpoint>();
                endpoint->serial = std::move(serial);
                endpoint->side = version->side;
                endpoint->version = version->version;
                endpoint->last_receive = now;
                endpoint->last_query = std::chrono::steady_clock::time_point::min();
                endpoints_[side] = std::move(endpoint);
                errors_[side].clear();
                occupied.insert(port);
                std::cout << "RealHandFFGGlovePlugin: connected " << side_name(side) << " glove " << version->version
                          << " on " << port << " at " << baudrate << " baud" << std::endl;
                break;
            }
            catch (const std::exception& error)
            {
                for (std::size_t side = 0; side < errors_.size(); ++side)
                {
                    if (!endpoints_[side])
                    {
                        errors_[side] = error.what();
                    }
                }
            }
        }
    }
}

void RealHandFFGGlovePlugin::drop_endpoint(std::size_t side, const std::string& error)
{
    std::cerr << "RealHandFFGGlovePlugin: disconnected " << side_name(side) << " glove: " << error << std::endl;
    endpoints_[side].reset();
    errors_[side] = error;
    last_discovery_ = std::chrono::steady_clock::time_point::min();
}

void RealHandFFGGlovePlugin::poll_endpoint(std::size_t side)
{
    auto& endpoint = endpoints_[side];
    if (!endpoint)
    {
        return;
    }
    try
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - endpoint->last_query >= kQueryPeriod)
        {
            endpoint->serial->send(Command::Position);
            endpoint->last_query = now;
        }
        for (const auto& frame : endpoint->serial->read_available(0))
        {
            endpoint->last_receive = now;
            if (auto positions = decode_positions(frame))
            {
                endpoint->positions = *positions;
                endpoint->has_positions = true;
                if (frame.command == Command::A6Position)
                {
                    // A6 firmware expects an A7 response before publishing the next sample.
                    endpoint->serial->send(Command::A7Force, std::vector<uint8_t>(kForceChannelCount * sizeof(float), 0));
                }
            }
        }
        if (now - endpoint->last_receive > kStaleTimeout)
        {
            drop_endpoint(side, "no serial response for one second");
            return;
        }
        if (endpoint->has_positions)
        {
            push_state(side);
        }
    }
    catch (const std::exception& error)
    {
        drop_endpoint(side, error.what());
    }
}

void RealHandFFGGlovePlugin::push_state(std::size_t side)
{
    core::JointStateOutputT output;
    output.device_id = std::string(kCollectionIds[side]);
    output.has_velocity = false;
    output.has_effort = false;
    output.ee_pose_valid = false;
    output.joints.reserve(kNumSensors);
    for (std::size_t sensor = 0; sensor < kNumSensors; ++sensor)
    {
        auto joint = std::make_shared<core::JointStateT>();
        joint->name = "sensor_" + std::to_string(sensor);
        joint->position = endpoints_[side]->positions[sensor];
        joint->valid = true;
        output.joints.push_back(std::move(joint));
    }

    const int64_t sample_time = core::os_monotonic_now_ns();
    flatbuffers::FlatBufferBuilder builder(kMaxFlatbufferSize);
    builder.Finish(core::JointStateOutput::Pack(builder, &output));
    pushers_[side]->push_buffer(builder.GetBufferPointer(), builder.GetSize(), sample_time, sample_time);
}

void RealHandFFGGlovePlugin::publish_status()
{
    std::vector<plugin_utils::PluginDeviceStatusEntry> entries;
    entries.reserve(2);
    for (std::size_t side = 0; side < endpoints_.size(); ++side)
    {
        plugin_utils::PluginDeviceStatusEntry entry{ .path = kDevicePaths[side] };
        if (!endpoints_[side])
        {
            entry.state = core::PluginDeviceState_DISCONNECTED;
            entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
            entry.error = errors_[side];
        }
        else if (!endpoints_[side]->has_positions)
        {
            entry.state = core::PluginDeviceState_DEGRADED;
            entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
            entry.error = "glove connected but no position frame has arrived";
        }
        else
        {
            entry.state = core::PluginDeviceState_CONNECTED;
            entry.reason = core::PluginDeviceReason_NONE;
        }
        entries.push_back(std::move(entry));
    }
    status_publisher_->publish_if_changed(entries, core::os_monotonic_now_ns());
}

void RealHandFFGGlovePlugin::update()
{
    discover();
    poll_endpoint(0);
    poll_endpoint(1);
    publish_status();
}

} // namespace plugins::realhand_ffg_glove
