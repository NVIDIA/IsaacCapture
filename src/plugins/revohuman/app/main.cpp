// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "revohuman_plugin.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/os_time.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

using namespace plugins::revohuman;

namespace
{

constexpr std::string_view kUsage =
    "Usage: revohuman_plugin\n"
    "         [--collection-prefix=NAME]\n"
    "         [--wrist-source=auto|hand_tracking|controller]\n"
    "         [--aim-to-wrist-left=px,py,pz,qx,qy,qz,qw] [--aim-to-wrist-right=px,py,pz,qx,qy,qz,qw]\n"
    "         [--plugin-root-id=ID]\n"
    "SIGINT or SIGTERM stops the plugin. SIGUSR1 tares the tactile sensors of the streaming gloves: keep the hands\n"
    "still and the tactile surfaces untouched for about 3 s.";

volatile std::sig_atomic_t g_stop = 0;
volatile std::sig_atomic_t g_tare = 0;

std::optional<std::string> flag_value(std::string_view arg, std::string_view name)
{
    if (arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '=')
        return std::string(arg.substr(name.size() + 1));
    return std::nullopt;
}

plugin_utils::WristSourceMode wrist_source_mode(const std::string& value, std::string_view name)
{
    if (value == "auto")
        return plugin_utils::WristSourceMode::Auto;
    if (value == "hand_tracking")
        return plugin_utils::WristSourceMode::HandTracking;
    if (value == "controller")
        return plugin_utils::WristSourceMode::Controller;
    throw std::invalid_argument(std::string(name) + " must be auto, hand_tracking or controller, got '" + value + "'");
}

// "px,py,pz,qx,qy,qz,qw" in meters, quaternion normalized.
XrPosef parse_pose(const std::string& text, std::string_view flag)
{
    std::array<double, 7> v{};
    std::size_t count = 0;
    bool ok = true;
    for (std::size_t begin = 0; ok && begin <= text.size(); ++count)
    {
        const std::size_t end = std::min(text.find(',', begin), text.size());
        const char* const item_end = text.data() + end;
        double value = 0;
        // std::from_chars skips no blanks, so an item must be the number alone.
        const auto [parsed_end, error] = std::from_chars(text.data() + begin, item_end, value);
        ok = count < v.size() && error == std::errc() && parsed_end == item_end && std::isfinite(value) &&
             std::abs(value) <= std::numeric_limits<float>::max();
        if (ok)
            v[count] = value;
        begin = end + 1;
    }
    const double n = std::sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6]);
    if (!ok || count != v.size() || !(n > 1e-6))
        throw std::invalid_argument(std::string(flag) + " needs seven finite numbers px,py,pz,qx,qy,qz,qw with a " +
                                    "non-zero quaternion, got '" + text + "'");
    return XrPosef{ { float(v[3] / n), float(v[4] / n), float(v[5] / n), float(v[6] / n) },
                    { float(v[0]), float(v[1]), float(v[2]) } };
}

} // namespace

int main(int argc, char** argv)
try
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.revohuman.main");
    RevoHumanPluginConfig config;
    std::optional<std::string> wrist_flag;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            std::cout << kUsage << std::endl; // Usage text is terminal UX, not a diagnostic.
            return 0;
        }
        if (auto value = flag_value(arg, "--collection-prefix"))
            config.collection_prefix = *value;
        else if (auto value = flag_value(arg, "--wrist-source"))
            config.wrist.mode = wrist_source_mode(*(wrist_flag = *value), "--wrist-source");
        else if (auto value = flag_value(arg, "--aim-to-wrist-left"))
            config.wrist.left_aim_to_wrist = parse_pose(*value, "--aim-to-wrist-left");
        else if (auto value = flag_value(arg, "--aim-to-wrist-right"))
            config.wrist.right_aim_to_wrist = parse_pose(*value, "--aim-to-wrist-right");
        else if (auto value = flag_value(arg, "--plugin-root-id"))
            config.plugin_root_id = *value;
        else
            logger->warn("ignoring unknown argument '{}'", arg);
    }
    if (config.collection_prefix.empty())
        throw std::invalid_argument("--collection-prefix must not be empty");
    if (config.plugin_root_id && config.plugin_root_id->empty())
        throw std::invalid_argument("--plugin-root-id must not be empty");
    // The hand capture labels a take with this variable, so a flag that disagrees is refused.
    if (const char* env = std::getenv("REVOHUMAN_WRIST_SOURCE"); env != nullptr && *env != '\0')
    {
        if (wrist_flag && *wrist_flag != env)
            throw std::invalid_argument("--wrist-source=" + *wrist_flag + " disagrees with REVOHUMAN_WRIST_SOURCE=" + env);
        config.wrist.mode = wrist_source_mode(env, "REVOHUMAN_WRIST_SOURCE");
    }

    std::signal(SIGINT, [](int) { g_stop = 1; });
    std::signal(SIGTERM, [](int) { g_stop = 1; });
    std::signal(SIGUSR1, [](int) { g_tare = 1; });
    RevoHumanPlugin plugin(config);
    logger->info("RevoHuman (collections: {}_<side>_<stream>)", config.collection_prefix);

    // Tick every millisecond, faster than any glove stream; report device status at 100 Hz.
    const auto start = std::chrono::steady_clock::now();
    bool reported_finish = false;
    for (std::size_t tick = 0; !g_stop; ++tick)
    {
        plugin.update(core::os_monotonic_now_ns());
        if (g_tare != 0)
        {
            g_tare = 0;
            logger->info("SIGUSR1: requesting a tactile tare");
            plugin.request_tare();
        }
        if (tick % 10 == 0)
            plugin.publish_status(core::os_monotonic_now_ns());
        if (plugin.finished() && !std::exchange(reported_finish, true))
            logger->info("both gloves given up; waiting for a stop signal"); // Status keeps showing why.
        std::this_thread::sleep_until(start + std::chrono::milliseconds(tick + 1));
    }
    logger->info("stopping");
    return 0;
}
catch (const std::exception& e)
{
    isaaccapture::Logger::get("isaaccapture.plugins.revohuman.main")->error("{}: {}", argv[0], e.what());
    return 1;
}
