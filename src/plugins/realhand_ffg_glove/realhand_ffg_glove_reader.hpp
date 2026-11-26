// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realhand_ffg_glove_serial.hpp"

#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace plugins::realhand_ffg_glove
{

class GloveReader
{
public:
    using Clock = std::chrono::steady_clock;
    struct Endpoint
    {
        std::unique_ptr<SerialGlove> serial;
        VersionInfo version;
        std::array<float, kNumSensors> positions{};
        bool has_positions = false;
        Clock::time_point last_receive{};
        std::optional<Clock::time_point> last_query;
    };

    explicit GloveReader(std::vector<std::string> ports = {}, std::vector<int> baudrates = {});
    void update(Clock::time_point now = Clock::now());
    const std::array<std::unique_ptr<Endpoint>, 2>& endpoints() const
    {
        return endpoints_;
    }
    const std::array<std::string, 2>& errors() const
    {
        return errors_;
    }

private:
    void discover(Clock::time_point now);
    void poll_endpoint(std::size_t side, Clock::time_point now);
    bool occupied(const std::string& port) const;
    void drop_endpoint(std::size_t side, const std::string& error);
    void advance_probe();

    std::vector<std::string> configured_ports_;
    std::vector<int> baudrates_;
    std::vector<std::string> candidates_;
    std::size_t candidate_index_ = 0;
    std::size_t baud_index_ = 0;
    std::optional<std::string> checked_port_;
    std::unique_ptr<SerialGlove> probe_;
    Clock::time_point probe_started_{};
    Clock::time_point probe_last_query_{};
    std::optional<Clock::time_point> last_discovery_;
    std::array<std::unique_ptr<Endpoint>, 2> endpoints_;
    std::array<std::string, 2> errors_;
};

} // namespace plugins::realhand_ffg_glove
