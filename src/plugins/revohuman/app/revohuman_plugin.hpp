// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_session/deviceio_session.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <plugin_utils/wrist_pose_source.hpp>
#include <revohuman_plugin/event_log.hpp>
#include <revohuman_plugin/live_source.hpp>
#include <revohuman_plugin/pipeline.hpp>
#include <revohuman_plugin/source.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace core
{
class OpenXRSession;
class XrTimeConverter;
} // namespace core

namespace plugins
{
namespace revohuman
{

struct RevoHumanPluginConfig
{
    //! Gloves through the RevoHuman C SDK. It finds the plugged-in gloves by itself, at most one per
    //! hand.
    LiveSourceConfig live;
    //! Collections are <prefix>_<side>_<stream>, e.g. revohuman_left_tactile.
    std::string collection_prefix = "revohuman";
    //! JSON-lines audit log of the run; disabled when empty.
    std::string event_log_path;
    PipelineConfig pipeline;
    //! Where the WRIST pose comes from: optical hand tracking, a controller, or both (Auto).
    plugin_utils::WristSourceMode wrist_source = plugin_utils::WristSourceMode::Auto;
    //! Controller aim pose to WRIST, indexed by Hand. Identity until a controller mount for the
    //! glove is calibrated.
    std::array<Pose, 2> aim_to_wrist{};
    //! Set by the plugin launcher; enables device status reporting for monitoring.
    std::optional<std::string> plugin_root_id;
};

/*!
 * @brief Pushes RevoHuman gloves into OpenXR: per hand an injected OpenXR hand (wrist from the
 * host's WristPoseSource), the encoder joint states, and the taxel forces with their TactileMetadata.
 */
std::unique_ptr<Source> make_live_source(const LiveSourceConfig& config);
//! Why this build's C SDK cannot make a LiveSource for the config, or nullopt when it can.
std::optional<std::string> live_source_unavailable_reason(const LiveSourceConfig& config);

//! TactileMetadata of a glove and where it came from.
struct TactileShape
{
    std::shared_ptr<const LoadedShape> shape;
    std::string source;
};
//! The TactileMetadata of @p hand, built from the RevoHuman C SDK's taxel layout; throws
//! std::runtime_error with the reason when the SDK has none (tactile is then not pushed).
TactileShape make_tactile_shape(Hand hand);

class RevoHumanPlugin
{
public:
    explicit RevoHumanPlugin(RevoHumanPluginConfig config);
    ~RevoHumanPlugin();

    RevoHumanPlugin(const RevoHumanPlugin&) = delete;
    RevoHumanPlugin& operator=(const RevoHumanPlugin&) = delete;

    //! Feeds what is due at now_ns and ticks both hands.
    void update(std::int64_t now_ns);
    //! Device status for monitoring; call at least once a second.
    void publish_status(std::int64_t now_ns);
    //! The source gives no more samples: both gloves were given up.
    bool finished() const
    {
        return finished_;
    }

private:
    class HandOutput;

    RevoHumanPluginConfig config_;
    class Wrists;

    plugin_utils::WristSourceConfig wrist_config_;
    plugin_utils::WristPoseSource::Requirements wrist_requirements_;
    std::shared_ptr<core::OpenXRSession> session_;
    std::unique_ptr<core::XrTimeConverter> time_converter_;
    std::unique_ptr<core::DeviceIOSession> deviceio_;
    std::unique_ptr<Wrists> wrists_;
    EventLog event_log_;
    std::array<std::unique_ptr<HandOutput>, 2> outputs_;
    std::array<std::unique_ptr<HandPipeline>, 2> pipelines_;
    std::unique_ptr<Source> source_;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> status_;
    bool finished_ = false;
};

} // namespace revohuman
} // namespace plugins
