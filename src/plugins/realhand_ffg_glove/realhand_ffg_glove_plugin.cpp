// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_plugin.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <schema/joint_state_generated.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>

namespace plugins::realhand_ffg_glove
{

namespace
{

constexpr std::size_t kMaxFlatbufferSize = 4096;
constexpr std::array<const char*, 2> kCollectionIds = { "realhand_ffg_glove_left", "realhand_ffg_glove_right" };
constexpr std::array<const char*, 2> kDevicePaths = { "/hand/left", "/hand/right" };
constexpr std::array<const char*, 2> kSensorPaths = { "/glove/realhand_ffg_glove_left",
                                                      "/glove/realhand_ffg_glove_right" };
constexpr auto kSkeletonStaleThreshold = std::chrono::milliseconds(200);

const char* side_name(std::size_t side)
{
    return side == 0 ? "left" : "right";
}

std::vector<std::string> required_extensions(const plugin_utils::WristPoseSource::Requirements& wrist)
{
    auto extensions = core::DeviceIOSession::get_required_extensions(wrist.trackers);
    extensions.push_back(XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME);
    for (const auto* extension : wrist.extensions)
    {
        if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end())
        {
            extensions.push_back(extension);
        }
    }
    for (const auto& extension : core::SchemaPusher::get_required_extensions())
    {
        if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end())
        {
            extensions.push_back(extension);
        }
    }
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
                                               std::string plugin_root_id,
                                               RealHandFFGGloveConfig config)
    : reader_(std::move(configured_ports), std::move(baudrates)), skeleton_(config.calibration_path, config.hand_scale)
{
    require_controller_wrist_ = config.wrist.mode == plugin_utils::WristSourceMode::Controller;
    const auto wrist = plugin_utils::WristPoseSource::collect_requirements(config.wrist.mode);
    session_ = std::make_shared<core::OpenXRSession>("RealHandFFGGlovePlugin", required_extensions(wrist));
    const auto handles = session_->get_handles();
    deviceio_session_ = core::DeviceIOSession::run(wrist.trackers, handles);
    time_converter_.emplace(handles);
    wrist_source_ = std::make_unique<plugin_utils::WristPoseSource>(
        config.wrist, handles, deviceio_session_.get(), wrist.controller_tracker);
    if (config.calibration_path.empty())
    {
        isaaccapture::Logger::get("isaaccapture.plugins.realhand_ffg_glove.RealHandFFGGlovePlugin")
            ->warn("No human calibration: raw sensors are available for calibration, but hands remain inactive");
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
}

RealHandFFGGlovePlugin::~RealHandFFGGlovePlugin() = default;

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
        joint->position = reader_.endpoints()[side]->positions[sensor];
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
    entries.reserve(4);
    for (std::size_t side = 0; side < reader_.endpoints().size(); ++side)
    {
        plugin_utils::PluginDeviceStatusEntry entry{ .path = kDevicePaths[side] };
        if (!reader_.endpoints()[side])
        {
            entry.state = core::PluginDeviceState_DISCONNECTED;
            entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
            entry.error = reader_.errors()[side];
        }
        else if (!reader_.endpoints()[side]->has_positions)
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
        auto sensor = entry;
        sensor.path = kSensorPaths[side];
        entries.push_back(std::move(sensor));
        if (entry.state == core::PluginDeviceState_CONNECTED && !injectors_[side])
        {
            entry.state = core::PluginDeviceState_DEGRADED;
            entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
            entry.error = wrist_rejected_[side] ?
                              "controller wrist unavailable or discontinuous; awaiting stable tracking" :
                          !skeleton_.calibrated(side == 0 ? HandSide::Left : HandSide::Right) ?
                              "missing human calibration for this hand" :
                              "no fresh usable skeleton";
        }
        entries.push_back(std::move(entry));
    }
    status_publisher_->publish_if_changed(entries, core::os_monotonic_now_ns());
}

void RealHandFFGGlovePlugin::update()
{
    deviceio_session_->update();
    reader_.update();
    const auto now = GloveReader::Clock::now();
    const auto time = time_converter_->os_monotonic_now();
    for (std::size_t side = 0; side < reader_.endpoints().size(); ++side)
    {
        const auto& endpoint = reader_.endpoints()[side];
        if (endpoint && endpoint->has_positions && now - endpoint->last_receive < kSkeletonStaleThreshold)
        {
            push_state(side);
            if (auto joints = skeleton_.reconstruct(endpoint->positions, endpoint->version.side))
            {
                const auto wrist = wrist_source_->query(side == 0, time);
                if (require_controller_wrist_)
                {
                    const bool rejected = !wrist_continuity_[side].accept(wrist, core::os_monotonic_now_ns());
                    if (rejected != wrist_rejected_[side])
                    {
                        auto logger =
                            isaaccapture::Logger::get("isaaccapture.plugins.realhand_ffg_glove.RealHandFFGGlovePlugin");
                        if (rejected)
                        {
                            logger->warn("{} controller wrist tracking interrupted; hand inactive, raw sensors retained",
                                         side_name(side));
                        }
                        else
                        {
                            logger->info("{} controller wrist tracking reacquired", side_name(side));
                        }
                    }
                    wrist_rejected_[side] = rejected;
                    if (rejected)
                    {
                        injectors_[side].reset();
                        continue;
                    }
                }
                *joints = place_hand(*joints, wrist.pose, wrist.valid, wrist.tracked);
                if (!injectors_[side])
                {
                    const auto handles = session_->get_handles();
                    injectors_[side] = std::make_unique<plugin_utils::HandInjector>(
                        handles.instance, handles.session, side == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT,
                        handles.space);
                }
                injectors_[side]->push(joints->data(), time);
                continue;
            }
        }
        // Destroying the injector makes isActive false; never leave a frozen hand behind.
        injectors_[side].reset();
    }
    publish_status();
}

} // namespace plugins::realhand_ffg_glove
