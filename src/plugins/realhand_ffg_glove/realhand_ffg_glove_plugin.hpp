// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realhand_ffg_glove_protocol.hpp"

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

class SerialGlove;

class RealHandFFGGlovePlugin
{
public:
    explicit RealHandFFGGlovePlugin(std::vector<std::string> configured_ports = {},
                                    std::vector<int> baudrates = {},
                                    std::string plugin_root_id = "realhand_ffg_glove");
    ~RealHandFFGGlovePlugin();

    void update();

private:
    struct Endpoint
    {
        std::unique_ptr<SerialGlove> serial;
        HandSide side = HandSide::Left;
        std::string version;
        std::array<float, kNumSensors> positions{};
        bool has_positions = false;
        std::chrono::steady_clock::time_point last_receive{};
        std::chrono::steady_clock::time_point last_query{};
    };

    void discover();
    void poll_endpoint(std::size_t side_index);
    void drop_endpoint(std::size_t side_index, const std::string& error);
    void push_state(std::size_t side_index);
    void publish_status();

    std::vector<std::string> configured_ports_;
    std::vector<int> baudrates_;
    std::array<std::unique_ptr<Endpoint>, 2> endpoints_;
    std::array<std::string, 2> errors_;
    std::chrono::steady_clock::time_point last_discovery_{};

    std::shared_ptr<core::OpenXRSession> session_;
    std::array<std::unique_ptr<core::SchemaPusher>, 2> pushers_;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> status_publisher_;
};

} // namespace plugins::realhand_ffg_glove
