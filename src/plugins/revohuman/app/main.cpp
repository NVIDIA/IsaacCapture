// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "revohuman_plugin.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/os_time.hpp>
#include <revohuman_plugin/pose_math.hpp>

#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

using namespace plugins::revohuman;

namespace
{

constexpr std::string_view kUsage =
    "Usage: revohuman_plugin\n"
    "         [--collection-prefix=NAME] [--event-log=FILE]\n"
    "         [--wrist-source=auto|hand_tracking|controller]\n"
    "         [--aim-to-wrist-left=px,py,pz,qx,qy,qz,qw] [--aim-to-wrist-right=px,py,pz,qx,qy,qz,qw]\n"
    "         [--stall-ms=N] [--plugin-root-id=ID]";

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int)
{
    g_stop = 1;
}

// The value of `--name=value` in @p arg, or nullopt when @p arg is another flag.
std::optional<std::string> flag_value(std::string_view arg, std::string_view name)
{
    if (arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '=')
        return std::string(arg.substr(name.size() + 1));
    return std::nullopt;
}

// The wrist source named @p value; @p name is the flag or variable it was given as.
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

std::int64_t ms_to_ns(const std::string& value, std::string_view flag)
{
    std::size_t used = 0;
    long long ms = -1;
    try
    {
        ms = std::stoll(value, &used);
    }
    catch (const std::exception&)
    {
        used = 0;
    }
    if (used != value.size() || ms <= 0)
    {
        throw std::invalid_argument(std::string(flag) + " needs a positive number of milliseconds, got '" + value + "'");
    }
    if (ms > std::numeric_limits<std::int64_t>::max() / 1'000'000)
    {
        throw std::invalid_argument(std::string(flag) + " is too large, got '" + value + "'");
    }
    return static_cast<std::int64_t>(ms) * 1'000'000;
}

} // namespace

int main(int argc, char** argv)
try
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.revohuman.main");

    RevoHumanPluginConfig config;
    std::array<bool, 2> aim_to_wrist_given{};
    std::optional<std::string> wrist_source_flag;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            // Usage text is terminal UX, not a diagnostic.
            std::cout << kUsage << std::endl;
            return 0;
        }
        if (auto value = flag_value(arg, "--collection-prefix"))
            config.collection_prefix = *value;
        else if (auto value = flag_value(arg, "--event-log"))
            config.event_log_path = *value;
        else if (auto value = flag_value(arg, "--wrist-source"))
        {
            config.wrist_source = wrist_source_mode(*value, "--wrist-source");
            wrist_source_flag = *value;
        }
        else if (auto value = flag_value(arg, "--aim-to-wrist-left"))
        {
            config.aim_to_wrist[static_cast<std::size_t>(Hand::Left)] = parse_pose(*value, "--aim-to-wrist-left");
            aim_to_wrist_given[static_cast<std::size_t>(Hand::Left)] = true;
        }
        else if (auto value = flag_value(arg, "--aim-to-wrist-right"))
        {
            config.aim_to_wrist[static_cast<std::size_t>(Hand::Right)] = parse_pose(*value, "--aim-to-wrist-right");
            aim_to_wrist_given[static_cast<std::size_t>(Hand::Right)] = true;
        }
        else if (auto value = flag_value(arg, "--stall-ms"))
            config.pipeline.stall_ns = ms_to_ns(*value, "--stall-ms");
        else if (auto value = flag_value(arg, "--plugin-root-id"))
            config.plugin_root_id = *value;
        else
            logger->warn("ignoring unknown argument '{}'", arg);
    }
    if (config.collection_prefix.empty())
        throw std::invalid_argument("--collection-prefix must not be empty");
    // The hand capture labels a take with this variable, so a flag that says otherwise is refused
    // rather than recorded under the wrong source.
    if (const char* env = std::getenv("REVOHUMAN_WRIST_SOURCE"); env != nullptr && *env != '\0')
    {
        const std::string value(env);
        if (wrist_source_flag && *wrist_source_flag != value)
            throw std::invalid_argument("--wrist-source=" + *wrist_source_flag +
                                        " disagrees with REVOHUMAN_WRIST_SOURCE=" + value);
        config.wrist_source = wrist_source_mode(value, "REVOHUMAN_WRIST_SOURCE");
    }
    // Before the OpenXR session is created, so a build whose SDK lacks the hand points or the product
    // hand models fails with this reason.
    if (const auto reason = live_source_unavailable_reason(config.live))
        throw std::invalid_argument(*reason);

    // The offset applies only while a hand's wrist comes from a controller: every frame with
    // `controller`, and with `auto` only when optical tracking has no pose for that hand.
    if (config.wrist_source != plugin_utils::WristSourceMode::HandTracking)
    {
        const char* when = config.wrist_source == plugin_utils::WristSourceMode::Controller ?
                               "WRIST is" :
                               "when optical tracking has no pose for this hand and a controller does, WRIST is";
        for (std::size_t i = 0; i < aim_to_wrist_given.size(); ++i)
            if (!aim_to_wrist_given[i])
                logger->warn(
                    "no --aim-to-wrist-{}: {} the controller aim pose with no offset", i == 0 ? "left" : "right", when);
    }

    logger->info("RevoHuman (collections: {}_<side>_<stream>)", config.collection_prefix);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    RevoHumanPlugin plugin(std::move(config));

    // Glove samples come at 200 Hz, so tick every millisecond and
    // report device status at 100 Hz.
    const auto tick = std::chrono::milliseconds(1);
    const auto program_start = std::chrono::steady_clock::now();
    std::size_t tick_count = 0;
    bool reported_finish = false;

    while (!g_stop)
    {
        const std::int64_t now_ns = core::os_monotonic_now_ns();
        plugin.update(now_ns);
        if (tick_count % 10 == 0)
            plugin.publish_status(core::os_monotonic_now_ns());
        if (plugin.finished() && !reported_finish)
        {
            // Keep the session up so status shows the gloves as gone rather than the plugin as dead.
            logger->info("source finished; waiting for a stop signal");
            reported_finish = true;
        }
        tick_count++;
        std::this_thread::sleep_until(program_start + tick * tick_count);
    }

    logger->info("stopping");
    return 0;
}
catch (const std::exception& e)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.revohuman.main");
    logger->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.revohuman.main");
    logger->error("{}: Unknown error", argv[0]);
    return 1;
}
