// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The Pose Provider plugin: runs the linked engine adapter through one PoseProvider run.

#include <log_bridge/logger.hpp>
#include <pose_engine_api/pose_engine.hpp>
#include <pose_provider/pose_provider.hpp>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

using namespace pose_provider;

namespace
{

constexpr const char* kLoggerName = "isaaccapture.plugins.pose_provider.main";
constexpr auto kIdlePeriod = std::chrono::milliseconds(5);

std::atomic<bool> g_finish_requested{ false };
static_assert(std::atomic<bool>::is_always_lock_free);

// A repeated signal must never escalate: plugins share the launcher's process group, so Ctrl+C reaches the plugin
// and PluginManager's stop then sends another SIGINT.
void on_signal(int)
{
    g_finish_requested.store(true);
}

struct Options
{
    ProviderConfig config;
    std::string plugin_root_id;
};

enum class ParseResult
{
    Ok,
    Help,
    Error,
};

std::string seconds_text(std::chrono::milliseconds duration)
{
    if (duration.count() % 1000 == 0)
    {
        return std::to_string(duration.count() / 1000);
    }
    std::string text = std::to_string(duration.count() / 1000) + ".";
    const std::string fraction = std::to_string(duration.count() % 1000);
    return text + std::string(3 - fraction.size(), '0') + fraction;
}

std::string usage()
{
    const ProviderConfig defaults{};
    return "Usage: pose_provider_plugin [--startup-timeout=<seconds>] [--drain-timeout=<seconds>] "
           "[--plugin-root-id=<id>] [-h|--help]\n"
           "  --startup-timeout=<seconds>  Time allowed from launch for the setup and the start (default " +
           seconds_text(defaults.startup_timeout) +
           ")\n"
           "  --drain-timeout=<seconds>    Time allowed from a finish request until the engine has drained (default " +
           seconds_text(defaults.drain_timeout) +
           ")\n"
           "  --plugin-root-id=<id>        Set by the plugin launcher\n"
           "  -h, --help                   Print this help\n"
           "Timeouts are between 0.001 and " +
           seconds_text(kMaxProviderTimeout) +
           " seconds; the startup timeout does not interrupt engine initialization. SIGINT or SIGTERM finishes the "
           "run.\n"
           "Exit status: 0 finished, 1 failed, 2 usage error.\n";
}

// Finite, 0 < seconds <= kMaxProviderTimeout, and at least 1 ms once rounded to milliseconds.
bool parse_seconds(std::string_view text, std::chrono::milliseconds& out)
{
    double seconds = 0.0;
    const char* end = text.data() + text.size();
    const auto [ptr, error] = std::from_chars(text.data(), end, seconds);
    if (error != std::errc{} || ptr != end || !std::isfinite(seconds) || seconds <= 0.0 ||
        seconds * 1000.0 > static_cast<double>(kMaxProviderTimeout.count()))
    {
        return false;
    }
    const std::chrono::milliseconds rounded{ std::llround(seconds * 1000.0) };
    if (rounded < std::chrono::milliseconds(1))
    {
        return false;
    }
    out = rounded;
    return true;
}

ParseResult parse_options(int argc, char** argv, Options& out, std::string& error)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            return ParseResult::Help;
        }
    }

    constexpr std::string_view kStartup = "--startup-timeout=";
    constexpr std::string_view kDrain = "--drain-timeout=";
    constexpr std::string_view kRootId = "--plugin-root-id=";
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg.starts_with(kStartup))
        {
            if (!parse_seconds(arg.substr(kStartup.size()), out.config.startup_timeout))
            {
                error = "invalid --startup-timeout '" + std::string(arg.substr(kStartup.size())) + "'";
                return ParseResult::Error;
            }
        }
        else if (arg.starts_with(kDrain))
        {
            if (!parse_seconds(arg.substr(kDrain.size()), out.config.drain_timeout))
            {
                error = "invalid --drain-timeout '" + std::string(arg.substr(kDrain.size())) + "'";
                return ParseResult::Error;
            }
        }
        else if (arg.starts_with(kRootId))
        {
            out.plugin_root_id = std::string(arg.substr(kRootId.size()));
        }
        else
        {
            for (const std::string_view option : { kStartup, kDrain, kRootId })
            {
                if (arg == option.substr(0, option.size() - 1))
                {
                    error = std::string(arg) + " takes its value in the form " + std::string(option) + "<value>";
                    return ParseResult::Error;
                }
            }
            error = "unknown argument '" + std::string(arg) + "'";
            return ParseResult::Error;
        }
    }
    return ParseResult::Ok;
}

} // namespace

int main(int argc, char** argv)
try
{
    Options options;
    std::string error;
    switch (parse_options(argc, argv, options, error))
    {
    case ParseResult::Help:
        std::cout << usage();
        return 0;
    case ParseResult::Error:
        // The rejection and the usage are one message, so they share one stream.
        std::cerr << "pose_provider_plugin: " << error << "\n\n" << usage();
        return 2;
    case ParseResult::Ok:
        break;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    auto engine = make_pose_engine();
    const EngineIdentity identity = engine->identity();
    isaaccapture::Logger::get(kLoggerName)
        ->info(
            "Pose Provider plugin: adapter {} {}, engine library {} {}; startup timeout {} ms, drain timeout {} "
            "ms; plugin root id '{}'",
            identity.adapter, identity.adapter_version, identity.engine_library, identity.engine_library_version,
            options.config.startup_timeout.count(), options.config.drain_timeout.count(), options.plugin_root_id);

    PoseProvider provider(options.config, std::move(engine));
    // Stands in for the application's start request until the plugin has a control channel.
    provider.request_start();
    while (!provider.finished())
    {
        if (g_finish_requested.load())
        {
            provider.request_finish();
        }
        provider.update();
        if (!provider.finished())
        {
            std::this_thread::sleep_for(kIdlePeriod);
        }
    }
    return provider.status().failed ? 1 : 0;
}
catch (const std::exception& e)
{
    isaaccapture::Logger::get(kLoggerName)->error("{}", e.what());
    return 1;
}
catch (...)
{
    isaaccapture::Logger::get(kLoggerName)->error("unknown exception");
    return 1;
}
