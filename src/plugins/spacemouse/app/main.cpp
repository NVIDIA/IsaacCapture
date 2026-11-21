// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spacemouse_plugin.hpp"

#include <log_bridge/logger.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

using namespace plugins::spacemouse;

namespace
{

constexpr std::string_view kUsage =
    "Usage: spacemouse_plugin [--device=/dev/hidrawN [--combined-report]] [--collection-id=ID] "
    "[--plugin-root-id=ID]";

// The value of `--name=value` in @p arg, or nullopt when @p arg is another flag.
std::optional<std::string> flag_value(std::string_view arg, std::string_view name)
{
    if (arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '=')
        return std::string(arg.substr(name.size() + 1));
    return std::nullopt;
}

} // namespace

int main(int argc, char** argv)
try
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.spacemouse.main");

    SpaceMousePluginConfig config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            // Usage text is terminal UX, not a diagnostic.
            std::cout << kUsage << std::endl;
            return 0;
        }
        if (auto value = flag_value(arg, "--device"))
            config.connection.device_path = *value;
        else if (arg == "--combined-report")
            config.connection.combined_report = true;
        else if (auto value = flag_value(arg, "--collection-id"))
            config.collection_id = *value;
        else if (auto value = flag_value(arg, "--plugin-root-id"))
            config.plugin_root_id = *value;
        else
            logger->warn("ignoring unknown argument '{}'", arg);
    }

    logger->info("SpaceMouse (device: {}, collection: {})",
                 config.connection.device_path ? config.connection.device_path->string() : "discovered",
                 config.collection_id);

    SpaceMousePlugin plugin(std::move(config));

    // Push data at 90 Hz
    const auto frame_duration = std::chrono::nanoseconds(1000000000 / 90);
    const auto program_start = std::chrono::steady_clock::now();
    std::size_t frame_count = 0;

    while (true)
    {
        plugin.update();
        frame_count++;
        std::this_thread::sleep_until(program_start + frame_duration * frame_count);
    }

    return 0;
}
catch (const std::exception& e)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.spacemouse.main");
    logger->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    auto logger = isaaccapture::Logger::get("isaaccapture.plugins.spacemouse.main");
    logger->error("{}: Unknown error", argv[0]);
    return 1;
}
