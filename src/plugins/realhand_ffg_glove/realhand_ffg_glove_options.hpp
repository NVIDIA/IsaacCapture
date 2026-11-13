// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realhand_ffg_glove_plugin.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace plugins::realhand_ffg_glove
{
struct PluginOptions
{
    std::vector<std::string> ports;
    std::vector<int> baudrates;
    std::string root_id = "realhand_ffg_glove";
    RealHandFFGGloveConfig config;
};

inline std::vector<std::string> split_csv(const std::string& value)
{
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (!item.empty())
        {
            result.push_back(item);
        }
    }
    return result;
}

inline std::vector<int> parse_baudrates(const std::string& value)
{
    std::vector<int> result;
    for (const auto& item : split_csv(value))
    {
        int baudrate = 0;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), baudrate);
        if (parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() ||
            std::find(kDefaultBaudrates.begin(), kDefaultBaudrates.end(), baudrate) == kDefaultBaudrates.end())
        {
            throw std::invalid_argument("--baudrates supports 2000000,1000000,921600,460800");
        }
        result.push_back(baudrate);
    }
    if (result.empty())
    {
        throw std::invalid_argument("--baudrates requires a supported baud rate");
    }
    return result;
}

inline plugin_utils::WristSourceMode parse_wrist_source(const std::string& source)
{
    if (source == "auto")
        return plugin_utils::WristSourceMode::Auto;
    if (source == "controller")
        return plugin_utils::WristSourceMode::Controller;
    if (source == "hand_tracking" || source == "handtracking")
        return plugin_utils::WristSourceMode::HandTracking;
    throw std::invalid_argument(
        "--wrist-source / REALHAND_FFG_GLOVE_WRIST_SOURCE must be auto, controller, or hand_tracking");
}

inline XrPosef parse_wrist_offset(const std::string& value)
{
    const auto fields = split_csv(value);
    if (fields.size() != 7)
    {
        throw std::invalid_argument("wrist offset must be px,py,pz,qx,qy,qz,qw (meters, XYZW)");
    }
    std::array<float, 7> numbers{};
    for (std::size_t i = 0; i < numbers.size(); ++i)
    {
        std::size_t consumed = 0;
        numbers[i] = std::stof(fields[i], &consumed);
        if (consumed != fields[i].size() || !std::isfinite(numbers[i]))
        {
            throw std::invalid_argument("wrist offset must contain finite numbers");
        }
    }
    const float norm =
        std::sqrt(numbers[3] * numbers[3] + numbers[4] * numbers[4] + numbers[5] * numbers[5] + numbers[6] * numbers[6]);
    if (!std::isfinite(norm) || norm < 1e-6f)
    {
        throw std::invalid_argument("wrist offset quaternion must have nonzero finite norm");
    }
    return { { numbers[3] / norm, numbers[4] / norm, numbers[5] / norm, numbers[6] / norm },
             { numbers[0], numbers[1], numbers[2] } };
}

inline PluginOptions parse_options(const std::vector<std::string>& arguments)
{
    PluginOptions options;
    if (const char* value = std::getenv("REALHAND_FFG_GLOVE_CALIBRATION"); value && *value)
        options.config.calibration_path = value;
    if (const char* value = std::getenv("REALHAND_FFG_GLOVE_PORTS"); value && *value)
        options.ports = split_csv(value);
    if (const char* value = std::getenv("REALHAND_FFG_GLOVE_WRIST_SOURCE"); value && *value)
        options.config.wrist.mode = parse_wrist_source(value);
    for (const auto& argument : arguments)
    {
        if (argument.starts_with("--ports="))
            options.ports = split_csv(argument.substr(8));
        else if (argument.starts_with("--baudrates="))
            options.baudrates = parse_baudrates(argument.substr(12));
        else if (argument.starts_with("--plugin-root-id="))
            options.root_id = argument.substr(17);
        else if (argument.starts_with("--calibration="))
            options.config.calibration_path = argument.substr(14);
        else if (argument.starts_with("--wrist-source="))
            options.config.wrist.mode = parse_wrist_source(argument.substr(15));
        else if (argument.starts_with("--left-aim-to-wrist="))
            options.config.wrist.left_aim_to_wrist = parse_wrist_offset(argument.substr(20));
        else if (argument.starts_with("--right-aim-to-wrist="))
            options.config.wrist.right_aim_to_wrist = parse_wrist_offset(argument.substr(21));
        else if (argument.starts_with("--hand-scale="))
        {
            const auto value = argument.substr(13);
            std::size_t consumed = 0;
            options.config.hand_scale = std::stof(value, &consumed);
            if (consumed != value.size() || !std::isfinite(options.config.hand_scale) || options.config.hand_scale <= 0)
                throw std::invalid_argument("--hand-scale must be finite and positive");
        }
        else
            throw std::invalid_argument("unknown argument: " + argument);
    }
    return options;
}
} // namespace plugins::realhand_ffg_glove
