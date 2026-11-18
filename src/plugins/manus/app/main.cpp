// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <manus/manus_hand_tracking_plugin.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace plugins::manus;

namespace
{

static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "lock-free atomic bool is required for signal safety");

std::atomic<bool> g_stop_requested{ false };

void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
    {
        g_stop_requested.store(true, std::memory_order_relaxed);
    }
}

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::vector<std::string> split_csv(const std::string& text)
{
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        // Trim whitespace
        const auto start = item.find_first_not_of(" \t");
        if (start == std::string::npos)
        {
            continue;
        }
        const auto end = item.find_last_not_of(" \t");
        out.push_back(item.substr(start, end - start + 1));
    }
    return out;
}

// Parses --{left,right}-aim-to-wrist=px,py,pz,qx,qy,qz,qw (meters, then quaternion).
XrPosef parse_aim_to_wrist(const std::string& arg, spdlog::logger& logger)
{
    const auto eq = arg.find('=');
    const std::string flag = arg.substr(0, eq);
    const std::string value = arg.substr(eq + 1);

    XrPosef pose{};
    int consumed = -1;

    const int parsed = std::sscanf(value.c_str(), "%f ,%f ,%f ,%f ,%f ,%f ,%f %n", &pose.position.x, &pose.position.y,
                                   &pose.position.z, &pose.orientation.x, &pose.orientation.y, &pose.orientation.z,
                                   &pose.orientation.w, &consumed);
    if (parsed != 7 || consumed < 0 || static_cast<std::size_t>(consumed) != value.size())
    {
        throw std::runtime_error("ManusHandPlugin: " + flag + " expects px,py,pz,qx,qy,qz,qw, got '" + value + "'");
    }

    const std::array<float, 7> components{ pose.position.x,    pose.position.y,    pose.position.z,
                                           pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                           pose.orientation.w };
    if (!std::all_of(components.begin(), components.end(), [](float c) { return std::isfinite(c); }))
    {
        throw std::runtime_error("ManusHandPlugin: " + flag + " values must be finite, got '" + value + "'");
    }

    XrQuaternionf& q = pose.orientation;
    const double norm = std::sqrt(static_cast<double>(q.x) * q.x + static_cast<double>(q.y) * q.y +
                                  static_cast<double>(q.z) * q.z + static_cast<double>(q.w) * q.w);
    if (norm < 1e-6f)
    {
        throw std::runtime_error("ManusHandPlugin: " + flag + " has a zero quaternion: '" + value + "'");
    }
    // Rounding keeps a unit quaternion's norm well inside this band; a value
    // written in XrPosef order (quaternion first) usually lands outside it.
    if (std::abs(norm - 1.0f) > 0.05f)
    {
        logger.warn("{}: quaternion norm is {:.3f}, normalizing it; check the order is px,py,pz,qx,qy,qz,qw", flag, norm);
    }
    q.x /= norm;
    q.y /= norm;
    q.z /= norm;
    q.w /= norm;

    logger.info("{} overrides the built-in controller-to-wrist offset: {},{},{},{},{},{},{}", flag, pose.position.x,
                pose.position.y, pose.position.z, q.x, q.y, q.z, q.w);
    return pose;
}

ManusPluginConfig parse_args(int argc, char** argv)
{
    ManusPluginConfig config;
    std::string datasets_arg = "human,sensors,haptic";
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.main");

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (starts_with(arg, "--datasets="))
        {
            datasets_arg = arg.substr(std::string("--datasets=").size());
        }
        else if (starts_with(arg, "--left-calibration="))
        {
            config.left_calibration_file = arg.substr(std::string("--left-calibration=").size());
        }
        else if (starts_with(arg, "--right-calibration="))
        {
            config.right_calibration_file = arg.substr(std::string("--right-calibration=").size());
        }
        else if (starts_with(arg, "--left-aim-to-wrist="))
        {
            config.left_aim_to_wrist = parse_aim_to_wrist(arg, *logger);
        }
        else if (starts_with(arg, "--right-aim-to-wrist="))
        {
            config.right_aim_to_wrist = parse_aim_to_wrist(arg, *logger);
        }
        else if (starts_with(arg, "--plugin-root-id="))
        {
            config.monitoring_plugin_root_id = arg.substr(std::string("--plugin-root-id=").size());
        }
        else
        {
            logger->warn("ignoring unknown argument '{}'", arg);
        }
    }

    config.human = false;
    config.sensors = false;
    config.haptic = false;
    for (const auto& ds : split_csv(datasets_arg))
    {
        if (ds == "human")
        {
            config.human = true;
        }
        else if (ds == "sensors")
        {
            config.sensors = true;
        }
        else if (ds == "haptic")
        {
            config.haptic = true;
        }
        else
        {
            logger->warn("ignoring unknown data set '{}'", ds);
        }
    }

    if (!config.human && !config.sensors && !config.haptic)
    {
        throw std::runtime_error("ManusHandPlugin: --datasets must enable at least one of human,sensors,haptic");
    }

    return config;
}

} // namespace

int main(int argc, char** argv)
try
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.main");
    logger->info("Manus Hand Plugin starting...");

    const ManusPluginConfig config = parse_args(argc, argv);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto& tracker = ManusTracker::instance(config);

    logger->info("Plugin running. Press Ctrl+C to stop.");

    // Target 90Hz frequency (~11.1ms period)
    const auto target_frame_duration = std::chrono::nanoseconds(1000000000 / 90);

    while (!g_stop_requested.load(std::memory_order_relaxed))
    {
        auto frame_start = std::chrono::steady_clock::now();

        tracker.update();

        std::this_thread::sleep_until(frame_start + target_frame_duration);
    }

    // Normal exit runs the singleton tracker's destructor and shuts down the SDK.
    return 0;
}
catch (const std::exception& e)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.main");
    logger->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.main");
    logger->error("{}: Unknown error occurred", argv[0]);
    return 1;
}
