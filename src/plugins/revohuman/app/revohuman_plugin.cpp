// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "revohuman_plugin.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/math.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <oxr_utils/pose_conversions.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <pusherio/schema_pusher.hpp>
#include <schema/tracker_metadata.hpp>

#include <XR_NVX1_device_interface.h>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>

namespace plugins
{
namespace revohuman
{
namespace
{

using plugin_utils::PluginDeviceStatusEntry;

// OpenXRSession enables the XrTimeConverter extensions itself; the status publisher's are enabled even without
// --plugin-root-id.
std::vector<std::string> required_extensions(const plugin_utils::WristPoseSource::Requirements& wrist)
{
    std::vector<std::string> out = core::DeviceIOSession::get_required_extensions(wrist.trackers);
    const auto add = [&out](const auto& more)
    {
        for (const auto& extension : more)
            if (std::find(out.begin(), out.end(), extension) == out.end())
                out.emplace_back(extension);
    };
    add(core::SchemaPusher::get_required_extensions());
    add(plugin_utils::PluginDeviceStatusPublisher::get_required_extensions());
    add(wrist.extensions);
    add(std::vector<std::string>{ XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME });
    return out;
}

core::SchemaPusherConfig pusher_config(const std::string& collection, std::size_t max_bytes, std::string tensor, Hand hand)
{
    return core::SchemaPusherConfig{ .collection_id = collection,
                                     .max_flatbuffer_size = max_bytes,
                                     .tensor_identifier = tensor,
                                     .localized_name = std::string("RevoHuman ") + hand_name(hand) + " " + tensor,
                                     .app_name = "RevoHumanPlugin" };
}

// A device that gave data before and stopped is DISCONNECTED; one that never did is UNKNOWN, so a plugin
// started before its gloves are plugged in is not reported as having lost them.
core::PluginDeviceState lost_or_unknown(bool seen)
{
    return seen ? core::PluginDeviceState_DISCONNECTED : core::PluginDeviceState_UNKNOWN;
}

// The /hand, /glove and /tactile rows of one hand, using only the state/reason pairs TeleopSession accepts
// (teleop_session_manager/status_monitor.py).
std::array<PluginDeviceStatusEntry, 3> hand_status(Hand hand, const HandHealth& h, const std::optional<GloveNote>& note)
{
    using namespace core;
    const std::string side = hand_name(hand);
    PluginDeviceStatusEntry hand_row{
        "/hand/" + side, lost_or_unknown(h.glove_seen), PluginDeviceReason_NO_HARDWARE_SIGNAL, {}
    };
    if (h.hand_active)
        hand_row =
            h.wrist_tracked ?
                PluginDeviceStatusEntry{ hand_row.path, PluginDeviceState_CONNECTED, PluginDeviceReason_NONE, {} } :
                PluginDeviceStatusEntry{ hand_row.path, PluginDeviceState_DEGRADED, PluginDeviceReason_NO_CURRENT_DATA,
                                         "no tracked wrist pose from the host" };

    PluginDeviceStatusEntry glove_row{
        "/glove/revohuman_" + side, PluginDeviceState_CONNECTED, PluginDeviceReason_NONE, {}
    };
    if (note && note->kind == GloveNote::Kind::Ended)
        glove_row = { glove_row.path, PluginDeviceState_FAILED, PluginDeviceReason_DEVICE_ERROR, note->text };
    else if (note && note->kind == GloveNote::Kind::Waiting)
        glove_row = { glove_row.path, lost_or_unknown(h.glove_seen), PluginDeviceReason_NO_HARDWARE_SIGNAL, note->text };
    else if (note && note->kind == GloveNote::Kind::NotReady)
        glove_row = { glove_row.path, h.glove_seen ? PluginDeviceState_DEGRADED : PluginDeviceState_UNKNOWN,
                      PluginDeviceReason_NO_CURRENT_DATA, note->text };
    else if (!h.glove_live)
        glove_row = { glove_row.path, lost_or_unknown(h.glove_seen), PluginDeviceReason_NO_HARDWARE_SIGNAL,
                      h.glove_seen ? "no recent glove sample" : "" };
    else if (note) // Unsynced: the samples flow, their timing against the other glove is degraded.
        glove_row = { glove_row.path, PluginDeviceState_DEGRADED, PluginDeviceReason_DEVICE_ERROR, note->text };

    PluginDeviceStatusEntry tactile_row{
        "/tactile/revohuman_" + side, lost_or_unknown(h.tactile_seen), PluginDeviceReason_NO_HARDWARE_SIGNAL, {}
    };
    if (h.tactile_live)
        tactile_row = { tactile_row.path, PluginDeviceState_CONNECTED, PluginDeviceReason_NONE, {} };
    else if (h.tactile_skip == TactileSkip::NoShape || h.tactile_skip == TactileSkip::NoTactile)
        tactile_row = { tactile_row.path, PluginDeviceState_DISABLED, PluginDeviceReason_DISABLED_BY_CONFIGURATION,
                        h.tactile_skip == TactileSkip::NoTactile ? "this glove has no tactile sensors" :
                                                                   "the C SDK has no taxel layout for this glove" };
    else if (h.glove_live && h.tactile_skip == TactileSkip::Uncalibrated)
        tactile_row = { tactile_row.path, PluginDeviceState_DEGRADED, PluginDeviceReason_CALIBRATION_FAILED,
                        "uncalibrated" };
    else if (h.glove_live && h.tactile_skip == TactileSkip::CountMismatch)
        tactile_row = { tactile_row.path, PluginDeviceState_FAILED, PluginDeviceReason_DEVICE_ERROR,
                        "taxel count differs from the TactileMetadata" };
    else if (h.glove_live && h.tactile_skip != TactileSkip::None)
        tactile_row = { tactile_row.path, PluginDeviceState_DEGRADED, PluginDeviceReason_NO_CURRENT_DATA,
                        tactile_skip_name(h.tactile_skip) };
    return { std::move(hand_row), std::move(glove_row), std::move(tactile_row) };
}

} // namespace

//! One hand's OpenXR outputs. The hand and the tactile collections exist only while the pipeline says so.
class RevoHumanPlugin::HandOutput
{
public:
    HandOutput(const core::OpenXRSessionHandles& handles,
               plugin_utils::WristPoseSource& wrists,
               Hand hand,
               const std::string& prefix)
        : handles_(handles),
          time_converter_(handles),
          wrists_(wrists),
          hand_(hand),
          collection_(prefix + "_" + hand_name(hand)),
          joint_state_(handles, pusher_config(collection_ + "_joint_state", kJointStateMaxBytes, "joint_state", hand))
    {
    }

    void push_joint_state(std::span<const std::uint8_t> buffer, std::int64_t sample_ns)
    {
        joint_state_.push_buffer(buffer.data(), buffer.size(), sample_ns, sample_ns);
    }
    void push_tactile(std::span<const std::uint8_t> buffer, std::int64_t sample_ns, std::int64_t raw_ns)
    {
        if (!tactile_)
            tactile_ = std::make_unique<core::SchemaPusher>(
                handles_, pusher_config(collection_ + "_tactile", kTactileMaxBytes, "tactile", hand_));
        tactile_->push_buffer(buffer.data(), buffer.size(), sample_ns, raw_ns);
    }
    void clear_tactile()
    {
        tactile_.reset();
    }
    void push_shape(std::span<const std::uint8_t> buffer, std::int64_t now_ns)
    {
        // Sized to the encoded table, so a glove with another layout gets a new collection.
        if (!shape_ || shape_size_ != buffer.size())
        {
            shape_.reset();
            shape_ = std::make_unique<core::SchemaPusher>(
                handles_, pusher_config(core::tracker_metadata_collection_id(collection_ + "_tactile"), buffer.size(),
                                        std::string(core::TRACKER_METADATA_TENSOR_IDENTIFIER), hand_));
            shape_size_ = buffer.size();
        }
        shape_->push_buffer(buffer.data(), buffer.size(), now_ns, now_ns);
    }
    void clear_shape()
    {
        shape_.reset();
    }
    void hand_create()
    {
        injector_ = std::make_unique<plugin_utils::HandInjector>(
            handles_.instance, handles_.session, hand_ == Hand::Left ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT,
            handles_.space);
    }
    void hand_destroy()
    {
        injector_.reset(); // Readers then see isActive = false.
    }
    bool hand_push(const HandJointPoses& poses, std::int64_t sample_ns)
    {
        const XrTime time = time_converter_.convert_monotonic_ns_to_xrtime(sample_ns);
        const plugin_utils::WristSample wrist = wrists_.query(hand_ == Hand::Left, time);
        const bool tracked = wrist.valid && wrist.tracked;
        std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
        for (std::size_t i = 0; i < joints.size(); ++i)
        {
            const XrPosef local = oxr_utils::to_xr_posef(poses.joints[i]);
            const bool position = (poses.position_valid_mask >> i) & 1u;
            const bool orientation = (poses.orientation_valid_mask >> i) & 1u;
            joints[i].locationFlags = (position ? XR_SPACE_LOCATION_POSITION_VALID_BIT : 0) |
                                      (orientation ? XR_SPACE_LOCATION_ORIENTATION_VALID_BIT : 0) |
                                      (position && tracked ? XR_SPACE_LOCATION_POSITION_TRACKED_BIT : 0) |
                                      (orientation && tracked ? XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT : 0);
            // Without a wrist pose the hand stays at the stage origin.
            joints[i].pose = wrist.valid ? oxr_utils::multiply_poses(wrist.pose, local) : local;
            joints[i].radius = 0.01f; // Nominal, for visualization.
        }
        assert(injector_ && "the pipeline creates the hand before it pushes");
        injector_->push(joints.data(), time);
        return tracked;
    }

private:
    core::OpenXRSessionHandles handles_;
    core::XrTimeConverter time_converter_;
    plugin_utils::WristPoseSource& wrists_;
    Hand hand_;
    std::string collection_;
    core::SchemaPusher joint_state_;
    std::unique_ptr<core::SchemaPusher> tactile_;
    std::unique_ptr<core::SchemaPusher> shape_;
    std::size_t shape_size_ = 0;
    std::unique_ptr<plugin_utils::HandInjector> injector_;
};

RevoHumanPlugin::RevoHumanPlugin(const RevoHumanPluginConfig& config)
{
    if (config.plugin_root_id && config.plugin_root_id->empty())
        throw std::invalid_argument("--plugin-root-id must not be empty");
    auto requirements = plugin_utils::WristPoseSource::collect_requirements(config.wrist.mode);
    session_ = std::make_shared<core::OpenXRSession>("RevoHumanPlugin", required_extensions(requirements));
    const auto handles = session_->get_handles();
    deviceio_ = core::DeviceIOSession::run(requirements.trackers, handles);
    wrists_ = std::make_unique<plugin_utils::WristPoseSource>(
        config.wrist, handles, deviceio_.get(), std::move(requirements.controller_tracker));
    for (const Hand hand : kHands)
    {
        auto& output = outputs_[index_of(hand)];
        output = std::make_unique<HandOutput>(handles, *wrists_, hand, config.collection_prefix);
        pipelines_[index_of(hand)] = std::make_unique<HandPipeline<HandOutput>>(hand, config.pipeline, *output);
    }
    source_ = std::make_unique<GloveSource>();
    if (config.plugin_root_id)
        status_ = std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(handles, *config.plugin_root_id);
}

RevoHumanPlugin::~RevoHumanPlugin() = default;

void RevoHumanPlugin::update(std::int64_t now_ns)
{
    deviceio_->update(); // WristPoseSource reads the controllers through this session, on this thread.
    source_->poll(now_ns, std::array{ pipelines_[0].get(), pipelines_[1].get() });
    for (auto& pipeline : pipelines_)
        pipeline->on_tick(now_ns);
}

void RevoHumanPlugin::publish_status(std::int64_t now_ns)
{
    if (!status_)
        return;
    std::vector<PluginDeviceStatusEntry> entries;
    for (const Hand hand : kHands)
        for (auto& row : hand_status(hand, pipelines_[index_of(hand)]->health(now_ns), source_->note(hand)))
            entries.push_back(std::move(row));
    status_->publish_if_changed(entries, now_ns);
}

} // namespace revohuman
} // namespace plugins
