// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realhand_ffg_glove_reader.hpp"
#include "realhand_ffg_glove_skeleton.hpp"

#include <deviceio_session/deviceio_session.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <plugin_utils/wrist_pose_source.hpp>
#include <pusherio/schema_pusher.hpp>

#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace core
{
class OpenXRSession;
}

namespace plugin_utils
{
class PluginDeviceStatusPublisher;
}

namespace plugins::realhand_ffg_glove
{

struct RealHandFFGGloveConfig
{
    std::string calibration_path;
    float hand_scale = 1.0f;
    // Mirrored PICO 4 Ultra controllers mounted upright on the FFG hand-back bracket.
    // Mount defaults are hardware configuration, separate from per-user finger calibration.
    plugin_utils::WristSourceConfig wrist{
        .left_aim_to_wrist = { { -0.422424918198f, 0.014580024139f, 0.024960358040f, 0.905936858676f },
                               { 0.020534770647f, -0.047368722071f, 0.192378854628f } },
        .right_aim_to_wrist = { { -0.422424918198f, -0.014580024139f, -0.024960358040f, 0.905936858676f },
                                { -0.020534770647f, -0.047368722071f, 0.192378854628f } }
    };
};

class RealHandFFGGlovePlugin
{
public:
    explicit RealHandFFGGlovePlugin(std::vector<std::string> configured_ports = {},
                                    std::vector<int> baudrates = {},
                                    std::string plugin_root_id = "realhand_ffg_glove",
                                    RealHandFFGGloveConfig config = {});
    ~RealHandFFGGlovePlugin();

    void update();

private:
    void push_state(std::size_t side_index);
    void publish_status();

    GloveReader reader_;
    HumanHandSkeleton skeleton_;

    std::shared_ptr<core::OpenXRSession> session_;
    std::shared_ptr<core::DeviceIOSession> deviceio_session_;
    std::unique_ptr<plugin_utils::WristPoseSource> wrist_source_;
    std::optional<core::XrTimeConverter> time_converter_;
    std::array<std::unique_ptr<plugin_utils::HandInjector>, 2> injectors_;
    std::array<std::unique_ptr<core::SchemaPusher>, 2> pushers_;
    std::array<WristPoseContinuity, 2> wrist_continuity_;
    std::array<bool, 2> wrist_rejected_{};
    bool require_controller_wrist_ = false;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> status_publisher_;
};

} // namespace plugins::realhand_ffg_glove
