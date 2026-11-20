// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera/camera_plugin.hpp"

#include "inc/camera/camera_host.hpp"
#include "inc/camera/config.hpp"
#include "inc/camera/endpoint_path.hpp"
#include "inc/camera/options.hpp"

#include <log_bridge/logger.hpp>

#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::camera
{

namespace
{

// How often the plugin logs what each camera did.
constexpr time_t ACTIVITY_LOG_SECONDS = 10;

// The running executable's own name, for the usage and the messages.
std::string program_name(int argc, char** argv, const Backend& backend)
{
    std::string name;
    if (argc > 0 && argv[0] != nullptr)
    {
        name = std::filesystem::path(argv[0]).filename().string();
    }
    return name.empty() ? "camera_" + backend.name : name;
}

std::string usage(const std::string& program, const Backend& backend)
{
    const std::string indent(std::string_view("Usage: ").size() + program.size() + 1, ' ');
    return "Usage: " + program + " [--config=FILE | --camera=ID[:KEY=VALUE,...] ... | --auto]\n" + indent +
           "[--endpoint=PATH | --instance=NAME] [--slots=N]\n"
           "\n"
           "The camera plugin of the '" +
           backend.name + "' backend: " + backend.description +
           ".\n"
           "It owns that backend's cameras and serves their frames to readers in other processes.\n"
           "Without arguments it serves the cameras its configuration file names for the backend;\n"
           "when there is no such file, every camera the backend finds plugged in.\n"
           "  --config=FILE     the cameras to serve (default: $XDG_CONFIG_HOME/isaaccapture/cameras.yaml,\n"
           "                    or ~/.config/isaaccapture/cameras.yaml); the cameras a file names for\n"
           "                    other backends are left to their plugins\n"
           "  --camera=...      one camera: the id readers ask for, then the backend's options for it;\n"
           "                    give it once per camera\n"
           "  --auto            serve every camera the backend finds plugged in, whatever a file says\n"
           "  --instance=NAME   the plugin's instance name (default: " +
           backend.name +
           "); it listens on\n"
           "                    camera-NAME.sock in this user's endpoint directory\n"
           "  --endpoint=PATH   the socket file readers connect to, instead of one in that directory\n"
           "  --slots=N         frames kept per stream (default 8); a reader with a larger window needs more\n";
}

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
    std::string instance;
    std::string config_file;
    std::vector<CameraConfig> cameras;
    std::optional<std::uint32_t> slots;
    bool automatic = false;
    bool help = false;
};

Arguments parse(int argc, char** argv, const Backend& backend)
{
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            arguments.help = true;
        }
        else if (arg == "--auto")
        {
            arguments.automatic = true;
        }
        else if (const auto endpoint = flag_value(arg, "--endpoint"))
        {
            arguments.endpoint = *endpoint;
        }
        else if (const auto instance = flag_value(arg, "--instance"))
        {
            arguments.instance = *instance;
        }
        else if (const auto config = flag_value(arg, "--config"))
        {
            arguments.config_file = *config;
        }
        else if (const auto camera = flag_value(arg, "--camera"))
        {
            arguments.cameras.push_back(parse_camera_argument(*camera, backend.name));
        }
        else if (const auto slots = flag_value(arg, "--slots"))
        {
            std::uint32_t count = 0;
            const auto [end, error] = std::from_chars(slots->data(), slots->data() + slots->size(), count);
            if (error != std::errc() || end != slots->data() + slots->size())
            {
                throw std::invalid_argument("--slots takes a whole number, not '" + std::string(*slots) + "'");
            }
            arguments.slots = count;
        }
        else if (flag_value(arg, "--plugin-root-id"))
        {
            // The plugin manager gives it to every plugin, as the root under which a plugin
            // reports the status of its devices. A camera plugin reports none.
        }
        else
        {
            throw std::invalid_argument("unknown argument '" + std::string(arg) + "'");
        }
    }
    if (!arguments.endpoint.empty() && !arguments.instance.empty())
    {
        throw std::invalid_argument("--endpoint and --instance both say where to listen; give one");
    }
    const int sources =
        (arguments.config_file.empty() ? 0 : 1) + (arguments.cameras.empty() ? 0 : 1) + (arguments.automatic ? 1 : 0);
    if (sources > 1)
    {
        throw std::invalid_argument("--config, --camera and --auto each say which cameras to serve; give one kind");
    }
    return arguments;
}

// The cameras to serve: from the command line, a file, or what is plugged in.
PluginConfig cameras_to_serve(const Arguments& arguments, const Backend& backend, spdlog::logger& logger)
{
    PluginConfig config;
    std::string source = "the command line";
    const std::string default_file = default_config_path();
    // A file says which cameras when one is named, or when the user's own is there and --auto
    // does not set it aside.
    const bool from_file =
        arguments.cameras.empty() &&
        (!arguments.config_file.empty() || (!arguments.automatic && std::filesystem::exists(default_file)));
    if (!arguments.cameras.empty())
    {
        config.cameras = arguments.cameras;
    }
    else if (from_file)
    {
        source = arguments.config_file.empty() ? default_file : arguments.config_file;
        config = load_config_file(source, backend.name);
    }
    else
    {
        source = "the cameras found plugged in";
        if (backend.discover)
        {
            config.cameras = backend.discover();
        }
    }
    if (arguments.slots)
    {
        config.slot_count = *arguments.slots;
    }
    if (config.cameras.empty())
    {
        // The advice is for where this run looked: a file it read is the one to fill in, --auto
        // reads none, and a camera plugged in is found only when no file is read.
        throw std::invalid_argument("there is no camera to serve in " + source + ". " +
                                    (arguments.automatic ? "Leave --auto out and name" : "Name") +
                                    " cameras with `backend: " + backend.name + "` in '" +
                                    (from_file ? source : default_file) + "' or with --camera" +
                                    (backend.discover && !from_file ? ", or plug one in" : ""));
    }
    logger.info("serving {} camera(s) from {}", config.cameras.size(), source);
    return config;
}

} // namespace

int run_camera_plugin(int argc, char** argv, const Backend& backend)
{
    const std::string program = program_name(argc, argv, backend);
    Arguments arguments;
    try
    {
        arguments = parse(argc, argv, backend);
    }
    catch (const std::exception& error)
    {
        // The refusal and the usage that follows are one message to the operator, on one channel.
        std::cerr << program << ": " << error.what() << "\n\n" << usage(program, backend);
        return 2;
    }
    if (arguments.help)
    {
        std::cout << usage(program, backend);
        return 0;
    }

    const auto logger = isaaccapture::Logger::get("isaaccapture.camera.run_camera_plugin");
    try
    {
        // Every thread started below inherits this mask, so the signals wait for the loop here.
        sigset_t stop_signals;
        sigemptyset(&stop_signals);
        sigaddset(&stop_signals, SIGINT);
        sigaddset(&stop_signals, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);

        const PluginConfig config = cameras_to_serve(arguments, backend, *logger);
        std::string endpoint = arguments.endpoint;
        if (endpoint.empty())
        {
            // A plugin's instance name is its backend's unless it was given another.
            endpoint = default_endpoint_path(arguments.instance.empty() ? backend.name : arguments.instance);
            prepare_endpoint_directory(endpoint);
        }

        CameraHost host(endpoint);
        // One call for all the cameras, so that what the SDK has open can be shared among them.
        std::vector<std::unique_ptr<CameraSource>> sources = backend.open(config.cameras);
        if (sources.size() != config.cameras.size())
        {
            throw std::logic_error("the '" + backend.name + "' backend was given " +
                                   std::to_string(config.cameras.size()) + " camera(s) and opened " +
                                   std::to_string(sources.size()));
        }
        for (std::unique_ptr<CameraSource>& source : sources)
        {
            host.add_camera(std::move(source), config.slot_count);
        }
        host.start();
        logger->info("{} serves {} camera(s) on '{}'", program, host.camera_count(), endpoint);

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
        logger->info("{} stops on signal {}", program, received);
        host.stop();
        return 0;
    }
    catch (const std::exception& error)
    {
        logger->error("{} cannot run: {}", program, error.what());
        return 1;
    }
}

} // namespace core::camera
