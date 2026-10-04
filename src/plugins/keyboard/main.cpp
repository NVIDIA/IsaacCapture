// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "keyboard_plugin.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace plugins::keyboard;

namespace
{

// Returns the first udev-classified keyboard event device, or nullopt if none found.
// Lets the plugin run with zero required arguments when launched via PluginManager
// (which invokes plugins as `<command> --plugin-root-id=<id>`, not positional args).
std::optional<std::string> discover_keyboard_device_path()
{
    const std::filesystem::path by_path_dir = "/dev/input/by-path";
    std::vector<std::string> candidates;
    std::error_code ec;
    if (!std::filesystem::exists(by_path_dir, ec))
        return std::nullopt;

    for (const auto& entry : std::filesystem::directory_iterator(by_path_dir, ec))
    {
        const std::string name = entry.path().filename().string();
        if (name.ends_with("-event-kbd"))
            candidates.push_back(entry.path().string());
    }
    if (candidates.empty())
        return std::nullopt;

    std::sort(candidates.begin(), candidates.end());
    return candidates.front();
}

// PluginManager invokes plugins as `<command> --plugin-root-id=<id> [plugin_args...]`.
// A bare positional token (no leading `--`) is treated as an explicit device path
// override, matching manual/standalone invocation.
struct ParsedArgs
{
    std::optional<std::string> device_path;
    std::string collection_id = "keyboard";
};

ParsedArgs parse_args(int argc, char** argv)
{
    ParsedArgs parsed;
    constexpr std::string_view kRootIdPrefix = "--plugin-root-id=";
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg.starts_with(kRootIdPrefix))
        {
            parsed.collection_id = std::string(arg.substr(kRootIdPrefix.size()));
        }
        else if (!arg.starts_with("--"))
        {
            parsed.device_path = std::string(arg);
        }
    }
    return parsed;
}

} // namespace

int main(int argc, char** argv)
try
{
    if (argc == 0)
    {
        std::cerr << "Usage: keyboard_plugin [device_path] [--plugin-root-id=<collection_id>]" << std::endl;
        return 1;
    }

    const ParsedArgs args = parse_args(argc, argv);
    std::optional<std::string> device_path = args.device_path;
    if (!device_path)
        device_path = discover_keyboard_device_path();
    if (!device_path)
    {
        std::cerr << argv[0] << ": No keyboard event device found under /dev/input/by-path/ and none given explicitly."
                  << std::endl;
        return 1;
    }

    std::cout << "Keyboard (device: " << *device_path << ", collection: " << args.collection_id << ")" << std::endl;

    KeyboardPlugin plugin(*device_path, args.collection_id);

    // Push data at 90 Hz.
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
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
catch (...)
{
    std::cerr << argv[0] << ": Unknown error" << std::endl;
    return 1;
}
