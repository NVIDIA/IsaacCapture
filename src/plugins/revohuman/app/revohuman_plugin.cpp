// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "revohuman_plugin.hpp"

#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <pusherio/schema_pusher.hpp>
#include <revohuman_plugin/device_status.hpp>
#include <revohuman_plugin/tactile.hpp>

#include <XR_NVX1_device_interface.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace plugins
{
namespace revohuman
{

namespace
{

constexpr std::array<Hand, 2> kHands = { Hand::Left, Hand::Right };

std::size_t index_of(Hand hand)
{
    return static_cast<std::size_t>(hand);
}

std::shared_ptr<spdlog::logger> plugin_logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.revohuman.RevoHumanPlugin");
}

void add_unique(std::vector<std::string>& extensions, const std::vector<std::string>& more)
{
    for (const auto& extension : more)
    {
        if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end())
        {
            extensions.push_back(extension);
        }
    }
}

std::vector<std::string> required_extensions(bool monitoring, const plugin_utils::WristPoseSource::Requirements& wrist)
{
    std::vector<std::string> extensions = core::DeviceIOSession::get_required_extensions(wrist.trackers);
    add_unique(extensions, core::SchemaPusher::get_required_extensions());
    for (const char* extension : wrist.extensions)
    {
        add_unique(extensions, { extension });
    }
    add_unique(extensions, core::XrTimeConverter::get_required_extensions());
    add_unique(extensions, { XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME });
    if (monitoring)
    {
        add_unique(extensions, plugin_utils::PluginDeviceStatusPublisher::get_required_extensions());
    }
    return extensions;
}

std::shared_ptr<core::OpenXRSession> make_session(const RevoHumanPluginConfig& config,
                                                  const plugin_utils::WristPoseSource::Requirements& wrist)
{
    if (config.plugin_root_id && config.plugin_root_id->empty())
    {
        throw std::invalid_argument("RevoHumanPlugin: --plugin-root-id must not be empty");
    }
    return std::make_shared<core::OpenXRSession>(
        "RevoHumanPlugin", required_extensions(config.plugin_root_id.has_value(), wrist));
}

core::SchemaPusherConfig pusher_config(const std::string& collection_id,
                                       std::size_t max_bytes,
                                       const char* tensor_identifier,
                                       const std::string& localized_name)
{
    return core::SchemaPusherConfig{ .collection_id = collection_id,
                                     .max_flatbuffer_size = max_bytes,
                                     .tensor_identifier = tensor_identifier,
                                     .localized_name = localized_name,
                                     .app_name = "RevoHumanPlugin" };
}

std::string side_name(Hand hand)
{
    return hand == Hand::Left ? "left" : "right";
}

plugin_utils::PluginDeviceStatusEntry entry(std::string path,
                                            core::PluginDeviceState state,
                                            core::PluginDeviceReason reason,
                                            std::string error = {})
{
    return plugin_utils::PluginDeviceStatusEntry{
        .path = std::move(path), .state = state, .reason = reason, .error = std::move(error)
    };
}

} // namespace

//! One hand's OpenXR outputs: three tensor collections and an injected hand. The hand and the two tactile
//! collections exist only while the pipeline says so.
class RevoHumanPlugin::HandOutput : public PushSink
{
public:
    HandOutput(const core::OpenXRSessionHandles& handles,
               const core::XrTimeConverter& time_converter,
               Hand hand,
               const std::string& prefix,
               std::size_t metadata_size)
        : handles_(handles),
          time_converter_(time_converter),
          hand_(hand),
          joint_state_(handles,
                       pusher_config(prefix + "_" + side_name(hand) + "_joint_state",
                                     kJointStateMaxBytes,
                                     "joint_state",
                                     "RevoHuman " + side_name(hand) + " joint state")),
          tactile_config_(pusher_config(prefix + "_" + side_name(hand) + "_tactile",
                                        kTactileMaxBytes,
                                        "tactile",
                                        "RevoHuman " + side_name(hand) + " tactile")),
          // The reader cap is an upper bound, not the size of every metadata tensor.
          shape_config_(pusher_config(core::tracker_metadata_collection_id(prefix + "_" + side_name(hand) + "_tactile"),
                                      metadata_size,
                                      core::TRACKER_METADATA_TENSOR_IDENTIFIER.data(),
                                      "RevoHuman " + side_name(hand) + " tactile metadata"))
    {
    }

    void push_joint_state(const std::vector<std::uint8_t>& buffer, std::int64_t sample_ns, std::int64_t raw_ns) override
    {
        joint_state_.push_buffer(buffer.data(), buffer.size(), sample_ns, raw_ns);
    }
    void push_tactile(const std::vector<std::uint8_t>& buffer, std::int64_t sample_ns, std::int64_t raw_ns) override
    {
        if (!tactile_)
            tactile_ = std::make_unique<core::SchemaPusher>(handles_, tactile_config_);
        tactile_->push_buffer(buffer.data(), buffer.size(), sample_ns, raw_ns);
    }
    void clear_tactile() override
    {
        tactile_.reset();
    }
    void push_shape(const std::vector<std::uint8_t>& buffer, std::int64_t now_ns) override
    {
        if (!shape_config_.max_flatbuffer_size)
            throw std::logic_error("metadata output was not configured");
        // Created with the first metadata, so a hand that never has a tactile glove has no tactile collections.
        if (!shape_)
            shape_ = std::make_unique<core::SchemaPusher>(handles_, shape_config_);
        shape_->push_buffer(buffer.data(), buffer.size(), now_ns, now_ns);
    }
    void clear_shape() override
    {
        shape_.reset();
    }
    void hand_create() override
    {
        injector_ = std::make_unique<plugin_utils::HandInjector>(
            handles_.instance, handles_.session, hand_ == Hand::Left ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT,
            handles_.space);
    }
    void hand_destroy() override
    {
        // Destroying the injector reports isActive = false to readers.
        injector_.reset();
    }
    void hand_push(const HandFrame& frame) override
    {
        if (!injector_)
        {
            return;
        }
        std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
        for (std::size_t i = 0; i < joints.size(); ++i)
        {
            const Pose& pose = frame.joints[i];
            const bool position_valid = (frame.position_valid_mask & (1u << i)) != 0;
            const bool orientation_valid = (frame.orientation_valid_mask & (1u << i)) != 0;
            joints[i].locationFlags =
                (position_valid ? XR_SPACE_LOCATION_POSITION_VALID_BIT : 0) |
                (orientation_valid ? XR_SPACE_LOCATION_ORIENTATION_VALID_BIT : 0) |
                (position_valid && frame.tracked ? XR_SPACE_LOCATION_POSITION_TRACKED_BIT : 0) |
                (orientation_valid && frame.tracked ? XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT : 0);
            joints[i].pose.position = XrVector3f{ static_cast<float>(pose.p[0]), static_cast<float>(pose.p[1]),
                                                  static_cast<float>(pose.p[2]) };
            joints[i].pose.orientation = XrQuaternionf{ static_cast<float>(pose.q[0]), static_cast<float>(pose.q[1]),
                                                        static_cast<float>(pose.q[2]), static_cast<float>(pose.q[3]) };
            joints[i].radius = frame.radius[i];
        }
        // Stamped with the glove sampling time, not the push time.
        injector_->push(joints.data(), time_converter_.convert_monotonic_ns_to_xrtime(frame.sample_mono_ns));
    }

private:
    core::OpenXRSessionHandles handles_;
    const core::XrTimeConverter& time_converter_;
    Hand hand_;
    core::SchemaPusher joint_state_;
    core::SchemaPusherConfig tactile_config_;
    core::SchemaPusherConfig shape_config_;
    std::unique_ptr<core::SchemaPusher> tactile_;
    std::unique_ptr<core::SchemaPusher> shape_;
    std::unique_ptr<plugin_utils::HandInjector> injector_;
};

XrPosef to_xr(const Pose& pose)
{
    return XrPosef{ { static_cast<float>(pose.q[0]), static_cast<float>(pose.q[1]), static_cast<float>(pose.q[2]),
                      static_cast<float>(pose.q[3]) },
                    { static_cast<float>(pose.p[0]), static_cast<float>(pose.p[1]), static_cast<float>(pose.p[2]) } };
}

plugin_utils::WristSourceConfig wrist_config(const RevoHumanPluginConfig& plugin)
{
    plugin_utils::WristSourceConfig config;
    config.mode = plugin.wrist_source;
    config.left_aim_to_wrist = to_xr(plugin.aim_to_wrist[index_of(Hand::Left)]);
    config.right_aim_to_wrist = to_xr(plugin.aim_to_wrist[index_of(Hand::Right)]);
    return config;
}

//! The host's wrist pose (WristPoseSource) at a glove sample time, in the stage.
class RevoHumanPlugin::Wrists : public WristProvider
{
public:
    Wrists(const plugin_utils::WristSourceConfig& config,
           const core::OpenXRSessionHandles& handles,
           core::DeviceIOSession* deviceio,
           std::shared_ptr<core::ControllerTracker> controller_tracker,
           const core::XrTimeConverter& time_converter)
        : source_(config, handles, deviceio, std::move(controller_tracker)), time_converter_(time_converter)
    {
    }

    WristPose wrist(Hand hand, std::int64_t sample_mono_ns) override
    {
        const plugin_utils::WristSample sample =
            source_.query(hand == Hand::Left, time_converter_.convert_monotonic_ns_to_xrtime(sample_mono_ns));
        WristPose out;
        out.valid = sample.valid;
        out.tracked = sample.tracked;
        out.pose_stage.p = { sample.pose.position.x, sample.pose.position.y, sample.pose.position.z };
        out.pose_stage.q = { sample.pose.orientation.x, sample.pose.orientation.y, sample.pose.orientation.z,
                             sample.pose.orientation.w };
        return out;
    }

private:
    plugin_utils::WristPoseSource source_;
    const core::XrTimeConverter& time_converter_;
};

RevoHumanPlugin::RevoHumanPlugin(RevoHumanPluginConfig config)
    : config_(std::move(config)),
      wrist_config_(wrist_config(config_)),
      wrist_requirements_(plugin_utils::WristPoseSource::collect_requirements(wrist_config_.mode)),
      session_(make_session(config_, wrist_requirements_)),
      time_converter_(std::make_unique<core::XrTimeConverter>(session_->get_handles())),
      event_log_(config_.event_log_path)
{
    auto logger = plugin_logger();
    const auto handles = session_->get_handles();
    deviceio_ = core::DeviceIOSession::run(wrist_requirements_.trackers, handles);
    wrists_ = std::make_unique<Wrists>(
        wrist_config_, handles, deviceio_.get(), std::move(wrist_requirements_.controller_tracker), *time_converter_);
    EventLog* log = event_log_.enabled() ? &event_log_ : nullptr;
    for (const Hand hand : kHands)
    {
        std::shared_ptr<const LoadedShape> shape;
        try
        {
            auto made = make_tactile_shape(hand);
            logger->info("{}: TactileMetadata from the {} ({} taxels, {} bytes, sha256 {})", hand_name(hand),
                         made.source, made.shape->taxel_count(), made.shape->bytes().size(), made.shape->sha256());
            shape = std::move(made.shape);
        }
        catch (const std::exception& e)
        {
            logger->warn("{}: no TactileMetadata ({}); tactile is not pushed", hand_name(hand), e.what());
        }
        auto& output = outputs_[index_of(hand)];
        output = std::make_unique<HandOutput>(
            handles, *time_converter_, hand, config_.collection_prefix, shape ? shape->bytes().size() : 0);
        auto& pipeline = pipelines_[index_of(hand)];
        pipeline = std::make_unique<HandPipeline>(hand, config_.pipeline, *output, log);
        pipeline->set_wrist_provider(wrists_.get());
        if (shape)
            pipeline->set_shape(std::move(shape));
    }
    source_ = make_live_source(config_.live);
    logger->info("live gloves");
    if (config_.plugin_root_id)
    {
        status_ = std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(handles, *config_.plugin_root_id);
    }
}

RevoHumanPlugin::~RevoHumanPlugin()
{
    // Remove the hands before the session goes away.
    pipelines_ = {};
    outputs_ = {};
    wrists_.reset();
    deviceio_.reset();
}

void RevoHumanPlugin::update(std::int64_t now_ns)
{
    // WristPoseSource reads the controllers through this session, on this thread.
    deviceio_->update();
    if (source_ && !finished_)
    {
        SourceBatch batch = source_->poll(now_ns);
        // SDK discovery and FK can take time; classify timestamps against the clock after receiving.
        now_ns = core::os_monotonic_now_ns();
        for (const Hand hand : kHands)
        {
            // A glove opened again starts over: new ticks, maybe the same generation.
            auto& pipeline = pipelines_[index_of(hand)];
            if (batch.hand_restarted[index_of(hand)] && pipeline)
            {
                pipeline->reset("glove_reopened");
            }
        }
        for (const SourceRecord& record : batch.records)
        {
            std::visit(
                [&](const auto& r)
                {
                    auto& pipeline = pipelines_[index_of(r.hand)];
                    if (!pipeline)
                    {
                        return;
                    }
                    using T = std::decay_t<decltype(r)>;
                    if constexpr (std::is_same_v<T, HandMeta>)
                    {
                        pipeline->set_meta(r);
                    }
                    else if constexpr (std::is_same_v<T, GloveSample>)
                    {
                        pipeline->on_glove(r, now_ns);
                    }
                },
                record);
        }
        if (batch.finished)
        {
            finished_ = true;
            plugin_logger()->info("live source finished after {} records", source_->records_released());
        }
    }
    for (auto& pipeline : pipelines_)
    {
        if (pipeline)
        {
            pipeline->on_tick(now_ns);
        }
    }
}

void RevoHumanPlugin::publish_status(std::int64_t now_ns)
{
    if (!status_)
    {
        return;
    }
    std::vector<plugin_utils::PluginDeviceStatusEntry> entries;
    for (const Hand hand : kHands)
    {
        HandStatusInput input;
        const auto& pipeline = pipelines_[index_of(hand)];
        if (pipeline)
        {
            input.health = pipeline->health(now_ns);
            input.note = source_->glove_note(hand);
        }
        for (DeviceStatusRow& row : make_hand_status(hand, input))
        {
            entries.push_back(entry(std::move(row.path), row.state, row.reason, std::move(row.error)));
        }
    }
    status_->publish_if_changed(entries, now_ns);
}

} // namespace revohuman
} // namespace plugins
