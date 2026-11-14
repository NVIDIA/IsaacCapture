// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_plugin/camera_host.hpp>
#include <camera_plugin/options.hpp>
#include <camera_plugin_backends/backends.hpp>
#include <log_bridge/logger.hpp>

#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace plugins::camera;

namespace
{

constexpr std::string_view USAGE =
    "Usage: camera_plugin --endpoint=PATH --camera=ID:BACKEND[:KEY=VALUE[,KEY=VALUE]...] [--camera=...]\n"
    "                     [--slots=N]\n"
    "       camera_plugin --list-backends\n"
    "\n"
    "Owns the cameras it is given and serves their frames to readers that connect to PATH.\n"
    "  --endpoint=PATH   the socket file readers connect to\n"
    "  --camera=...      one camera: the id readers ask for, the backend that opens it, and the\n"
    "                    backend's options; give it once per camera\n"
    "  --slots=N         frames kept per stream (default 8); a reader with a larger window needs more\n"
    "  --list-backends   print the backends this build has and exit\n";

// How often the plugin logs what each camera did.
constexpr time_t ACTIVITY_LOG_SECONDS = 10;

// The value of `--name=value` in `arg`, or nothing when `arg` is another flag.
std::optional<std::string_view> flag_value(std::string_view arg, std::string_view name)
{
    if (arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '=')
    {
        return arg.substr(name.size() + 1);
    }
    return std::nullopt;
}

struct Arguments
{
    std::string endpoint;
    std::vector<CameraConfig> cameras;
    std::uint32_t slots = DEFAULT_SLOT_COUNT;
    bool list_backends = false;
    bool help = false;
};

Arguments parse(int argc, char** argv)
{
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            arguments.help = true;
        }
        else if (arg == "--list-backends")
        {
            arguments.list_backends = true;
        }
        else if (const auto endpoint = flag_value(arg, "--endpoint"))
        {
            arguments.endpoint = *endpoint;
        }
        else if (const auto camera = flag_value(arg, "--camera"))
        {
            arguments.cameras.push_back(parse_camera_argument(*camera));
        }
        else if (const auto slots = flag_value(arg, "--slots"))
        {
            const auto [end, error] = std::from_chars(slots->data(), slots->data() + slots->size(), arguments.slots);
            if (error != std::errc() || end != slots->data() + slots->size())
            {
                throw std::invalid_argument("--slots takes a whole number, not '" + std::string(*slots) + "'");
            }
        }
        else if (flag_value(arg, "--plugin-root-id"))
        {
            // Added by the plugin manager for plugins with a device tree; cameras have none.
        }
        else
        {
            throw std::invalid_argument("unknown argument '" + std::string(arg) + "'");
        }
    }
    if (!arguments.help && !arguments.list_backends)
    {
        if (arguments.endpoint.empty())
        {
            throw std::invalid_argument("--endpoint is required");
        }
        if (arguments.cameras.empty())
        {
            throw std::invalid_argument("at least one --camera is required");
        }
    }
    return arguments;
}

} // namespace

int main(int argc, char** argv)
{
    Arguments arguments;
    try
    {
        arguments = parse(argc, argv);
    }
    catch (const std::exception& error)
    {
        // The refusal and the usage that follows are one message to the operator, on one channel.
        std::cerr << "camera_plugin: " << error.what() << "\n\n" << USAGE;
        return 2;
    }
    const std::vector<Backend> backends = built_backends();
    if (arguments.help)
    {
        std::cout << USAGE;
        return 0;
    }
    if (arguments.list_backends)
    {
        for (const Backend& backend : backends)
        {
            std::cout << backend.name << ": " << backend.description << "\n";
        }
        return 0;
    }

    const auto logger = isaaccapture::Logger::get("isaaccapture.plugins.camera.main");
    try
    {
        // Every thread started below inherits this mask, so the signals wait for sigwait() here.
        sigset_t stop_signals;
        sigemptyset(&stop_signals);
        sigaddset(&stop_signals, SIGINT);
        sigaddset(&stop_signals, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);

        CameraHost host(arguments.endpoint);
        for (const CameraConfig& camera : arguments.cameras)
        {
            host.add_camera(find_backend(backends, camera.backend).open(camera), arguments.slots);
        }
        host.start();
        logger->info("camera plugin serves {} camera(s) on '{}'", host.camera_count(), arguments.endpoint);

        // Sleeps until a signal says stop, waking now and then to log what the cameras did.
        const timespec activity_interval = { ACTIVITY_LOG_SECONDS, 0 };
        int received = 0;
        while ((received = sigtimedwait(&stop_signals, nullptr, &activity_interval)) <= 0)
        {
            if (errno == EAGAIN)
            {
                host.log_activity();
            }
        }
        logger->info("camera plugin stops on signal {}", received);
        host.stop();
        return 0;
    }
    catch (const std::exception& error)
    {
        logger->error("camera plugin cannot run: {}", error.what());
        return 1;
    }
}
