// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_plugin.hpp"

#include <log_bridge/logger.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

std::atomic<bool> g_stop_requested{ false };

void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
    {
        g_stop_requested.store(true, std::memory_order_relaxed);
    }
}

std::vector<std::string> split_csv(const std::string& value)
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

std::vector<int> parse_baudrates(const std::string& value)
{
    std::vector<int> result;
    for (const auto& item : split_csv(value))
    {
        result.push_back(std::stoi(item));
    }
    return result;
}

XrPosef parse_pose(const std::string& value)
{
    const auto fields = split_csv(value);
    if (fields.size() != 7)
    {
        throw std::invalid_argument("wrist offset must be px,py,pz,qx,qy,qz,qw (meters, XYZW)");
    }
    std::array<float, 7> v{};
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        std::size_t consumed = 0;
        v[i] = std::stof(fields[i], &consumed);
        if (consumed != fields[i].size() || !std::isfinite(v[i]))
        {
            throw std::invalid_argument("wrist offset must contain finite numbers");
        }
    }
    const float norm = std::sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6]);
    if (!std::isfinite(norm) || norm < 1e-6f)
    {
        throw std::invalid_argument("wrist offset quaternion must have nonzero finite norm");
    }
    return { { v[3] / norm, v[4] / norm, v[5] / norm, v[6] / norm }, { v[0], v[1], v[2] } };
}

} // namespace

int main(int argc, char** argv)
try
{
    std::vector<std::string> ports;
    std::vector<int> baudrates;
    std::string plugin_root_id = "realhand_ffg_glove";
    plugins::realhand_ffg_glove::RealHandFFGGloveConfig config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument.starts_with("--ports="))
        {
            ports = split_csv(argument.substr(8));
        }
        else if (argument.starts_with("--baudrates="))
        {
            baudrates = parse_baudrates(argument.substr(12));
        }
        else if (argument.starts_with("--plugin-root-id="))
        {
            plugin_root_id = argument.substr(17);
        }
        else if (argument.starts_with("--calibration="))
        {
            config.calibration_path = argument.substr(14);
        }
        else if (argument.starts_with("--hand-scale="))
        {
            const auto value = argument.substr(13);
            std::size_t consumed = 0;
            config.hand_scale = std::stof(value, &consumed);
            if (consumed != value.size())
            {
                throw std::invalid_argument("hand scale must be a number");
            }
        }
        else if (argument.starts_with("--left-aim-to-wrist="))
        {
            config.wrist.left_aim_to_wrist = parse_pose(argument.substr(20));
        }
        else if (argument.starts_with("--right-aim-to-wrist="))
        {
            config.wrist.right_aim_to_wrist = parse_pose(argument.substr(21));
        }
        else if (argument.starts_with("--wrist-source="))
        {
            const auto source = argument.substr(15);
            if (source == "auto")
            {
                config.wrist.mode = plugin_utils::WristSourceMode::Auto;
            }
            else if (source == "controller")
            {
                config.wrist.mode = plugin_utils::WristSourceMode::Controller;
            }
            else if (source == "handtracking")
            {
                config.wrist.mode = plugin_utils::WristSourceMode::HandTracking;
            }
            else
            {
                throw std::invalid_argument("wrist source must be auto, controller, or handtracking");
            }
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    plugins::realhand_ffg_glove::RealHandFFGGlovePlugin plugin(
        std::move(ports), std::move(baudrates), plugin_root_id, std::move(config));
    isaaccapture::Logger::get("isaaccapture.plugins.realhand_ffg_glove.main")->info("RealHand FFG Glove plugin running");

    constexpr auto frame_period = std::chrono::nanoseconds(1'000'000'000 / 90);
    while (!g_stop_requested.load(std::memory_order_relaxed))
    {
        const auto frame_start = std::chrono::steady_clock::now();
        plugin.update();
        std::this_thread::sleep_until(frame_start + frame_period);
    }
    return 0;
}
catch (const std::exception& error)
{
    isaaccapture::Logger::get("isaaccapture.plugins.realhand_ffg_glove.main")->error("{}: {}", argv[0], error.what());
    return 1;
}
