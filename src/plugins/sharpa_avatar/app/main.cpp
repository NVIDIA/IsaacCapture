// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Avatar hand plugin entrypoint. Runs a fixed-rate loop that pumps the
// AvatarTracker: HUMAN → OpenXR hand_tracker, RAW/ROBOT → JointState tensors,
// and optional haptic commands → Avatar vibration motors.
//
// Usage:
//   ./avatar_hand_plugin [sdk_config.json] [--datasets=human,raw,robot,haptic]
//
// The CloudXR runtime must be running and its environment sourced first:
//   python -m isaacteleop.cloudxr
//   source ~/.cloudxr/run/cloudxr.env

#include <avatar/avatar_hand_tracking_plugin.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace plugins::avatar;

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

AvatarPluginConfig parse_args(int argc, char** argv)
{
    AvatarPluginConfig config;
    std::string datasets_arg = "human,raw,robot,haptic";

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (starts_with(arg, "--datasets="))
        {
            datasets_arg = arg.substr(std::string("--datasets=").size());
        }
        else if (starts_with(arg, "--plugin-root-id="))
        {
            // Injected by the PluginManager; unused by this plugin.
        }
        else if (!starts_with(arg, "--"))
        {
            config.sdk_config_path = arg;
        }
        else
        {
            std::cerr << "AvatarHandPlugin: ignoring unknown argument '" << arg << "'" << std::endl;
        }
    }

    config.human = false;
    config.raw = false;
    config.robot = false;
    config.haptic = false;
    for (const auto& ds : split_csv(datasets_arg))
    {
        if (ds == "human")
        {
            config.human = true;
        }
        else if (ds == "raw")
        {
            config.raw = true;
        }
        else if (ds == "robot")
        {
            config.robot = true;
        }
        else if (ds == "haptic")
        {
            config.haptic = true;
        }
        else
        {
            std::cerr << "AvatarHandPlugin: ignoring unknown data set '" << ds << "'" << std::endl;
        }
    }

    if (!config.human && !config.raw && !config.robot && !config.haptic)
    {
        throw std::runtime_error("AvatarHandPlugin: --datasets must enable at least one of human,raw,robot,haptic");
    }

    // Supplied by plugin.yaml's `args`. Required rather than defaulted: the file
    // defines human landmark order and the raw/robot joint names, so a wrong or
    // missing path has to fail loudly instead of degrading to an empty mapping.
    if (config.sdk_config_path.empty())
    {
        throw std::runtime_error("AvatarHandPlugin: no sdk_config.json path given; set it in plugin.yaml's args");
    }

    return config;
}

} // namespace

int main(int argc, char** argv)
try
{
    std::cout << "Avatar Hand Plugin starting..." << std::endl;

    const AvatarPluginConfig config = parse_args(argc, argv);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    AvatarTracker tracker(config);

    std::cout << "Plugin running. Press Ctrl+C to stop." << std::endl;

    // Target 90Hz frequency (~11.1ms period).
    const auto target_frame_duration = std::chrono::nanoseconds(1000000000 / 90);

    while (!g_stop_requested.load(std::memory_order_relaxed))
    {
        const auto frame_start = std::chrono::steady_clock::now();

        tracker.update();

        std::this_thread::sleep_until(frame_start + target_frame_duration);
    }

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
catch (...)
{
    std::cerr << argv[0] << ": Unknown error occurred" << std::endl;
    return 1;
}
