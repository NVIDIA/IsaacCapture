// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "glove_source.hpp"

#include <deviceio_session/deviceio_session.hpp>
#include <oxr/oxr_session.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <plugin_utils/wrist_pose_source.hpp>
#include <revohuman_plugin/pipeline.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace plugins
{
namespace revohuman
{

struct RevoHumanPluginConfig
{
    //! Collections are <prefix>_<side>_<stream>, e.g. revohuman_left_tactile.
    std::string collection_prefix = "revohuman";
    PipelineConfig pipeline;
    //! The aim-to-wrist offsets end at the glove's WRIST; identity until a glove mount is calibrated.
    plugin_utils::WristSourceConfig wrist;
    //! Set by the plugin launcher; enables device status reporting.
    std::optional<std::string> plugin_root_id;
};

//! Pushes RevoHuman gloves into OpenXR: per hand an injected hand, the encoder joint states, and the taxel
//! forces with their TactileMetadata.
class RevoHumanPlugin
{
public:
    explicit RevoHumanPlugin(const RevoHumanPluginConfig& config);
    ~RevoHumanPlugin();

    void update(std::int64_t now_ns);
    //! Call at least once a second.
    void publish_status(std::int64_t now_ns);
    bool finished() const
    {
        return source_->finished();
    }
    //! See GloveSource::request_tare().
    void request_tare()
    {
        source_->request_tare();
    }

private:
    class HandOutput;

    // Destroyed in reverse order: the hands go before the session.
    std::shared_ptr<core::OpenXRSession> session_;
    std::unique_ptr<core::DeviceIOSession> deviceio_;
    std::unique_ptr<plugin_utils::WristPoseSource> wrists_;
    std::array<std::unique_ptr<HandOutput>, 2> outputs_;
    std::array<std::unique_ptr<HandPipeline<HandOutput>>, 2> pipelines_;
    std::unique_ptr<GloveSource> source_;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> status_;
};

} // namespace revohuman
} // namespace plugins
