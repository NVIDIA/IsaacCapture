// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <log_bridge/logger.hpp>
#include <manus/manus_hand_tracking_plugin.hpp>
#if defined(MANUS_HAVE_REMOTE_PLUGIN_SESSION)
#    include <remote_plugin_session/bridge_limits.hpp>
#    include <remote_plugin_session/remote_address.hpp>
#    include <remote_plugin_session/remote_plugin_session.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <memory>
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

struct AppArgs
{
    ManusPluginConfig config;
    //! host[:port] or ws://host[:port][/path] of a remote_device_bridge; empty selects the local OpenXR session.
    std::string remote_endpoint;
};

AppArgs parse_args(int argc, char** argv)
{
    AppArgs args;
    ManusPluginConfig& config = args.config;
    std::string datasets_arg = "human,sensors,haptic";
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.main");
    bool datasets_explicit = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (starts_with(arg, "--datasets="))
        {
            datasets_arg = arg.substr(std::string("--datasets=").size());
            datasets_explicit = true;
        }
        else if (starts_with(arg, "--remote="))
        {
            args.remote_endpoint = arg.substr(std::string("--remote=").size());
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

    // RemotePluginSession's pull channel exposes no trackers, so the haptic reader cannot run over --remote.
    if (!args.remote_endpoint.empty() && config.haptic)
    {
        if (datasets_explicit)
        {
            throw std::invalid_argument(
                "ManusHandPlugin: --remote cannot be combined with the 'haptic' data set; "
                "use --datasets=human,sensors");
        }
        config.haptic = false;
        logger->info("ManusHandPlugin: --remote given; defaulting to --datasets=human,sensors");
    }

    return args;
}

} // namespace

int main(int argc, char** argv)
try
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.main");
    logger->info("Manus Hand Plugin starting...");

    const AppArgs args = parse_args(argc, argv);
    const ManusPluginConfig& config = args.config;
#if !defined(MANUS_HAVE_REMOTE_PLUGIN_SESSION)
    if (!args.remote_endpoint.empty())
    {
        throw std::invalid_argument(
            "ManusHandPlugin: --remote requires a build with BUILD_PLUGIN_REMOTE_DEVICE_BRIDGE=ON");
    }
#endif

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Empty for the local path: ManusTracker then opens its own OpenXR session, so this executable links no loader.
    core::PluginSessionFactory remote_session_factory;
#if defined(MANUS_HAVE_REMOTE_PLUGIN_SESSION)
    if (!args.remote_endpoint.empty())
    {
        // parse_args() turns haptic off for --remote, so ManusTracker passes no trackers here.
        remote_session_factory = [remote = args.remote_endpoint](
                                     const core::PluginSessionRequirements& requirements,
                                     std::vector<std::shared_ptr<core::ITracker>>) -> std::unique_ptr<core::IPluginSession>
        {
            isaaccapture::Logger::get("isaaccapture.plugins.manus.main")->info("Using remote plugin session at {}", remote);
            return std::make_unique<remote_plugin_session::RemotePluginSession>(
                remote_plugin_session::connect_transport(
                    remote_plugin_session::parse_remote(remote, remote_plugin_session::kDefaultBridgePort)),
                "ManusHandPlugin", requirements);
        };
    }
#endif

    auto& tracker = remote_session_factory ? ManusTracker::instance(config, remote_session_factory) :
                                             ManusTracker::instance(config);

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
