// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_ipc/endpoint_path.hpp>
#include <camera_plugin/camera_host.hpp>
#include <camera_plugin/config.hpp>
#include <camera_plugin/options.hpp>
#include <camera_plugin_backends/backends.hpp>
#include <log_bridge/logger.hpp>

#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
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
    "Usage: camera_plugin [--config=FILE | --camera=ID:BACKEND[:KEY=VALUE,...] ... | --auto]\n"
    "                     [--endpoint=PATH | --instance=NAME] [--slots=N]\n"
    "       camera_plugin --list-backends\n"
    "\n"
    "Owns cameras and serves their frames to readers in other processes. Without arguments it\n"
    "serves the cameras of its configuration file on this user's default endpoint; when there is\n"
    "no such file, every camera its backends find plugged in.\n"
    "  --config=FILE     the cameras to serve (default: $XDG_CONFIG_HOME/isaaccapture/cameras.yaml,\n"
    "                    or ~/.config/isaaccapture/cameras.yaml)\n"
    "  --camera=...      one camera: the id readers ask for, the backend that opens it, and the\n"
    "                    backend's options; give it once per camera\n"
    "  --auto            serve every camera the backends find plugged in, whatever a file says\n"
    "  --instance=NAME   run as a named instance beside the default one, on its own endpoint\n"
    "  --endpoint=PATH   the socket file readers connect to, instead of a default endpoint\n"
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
    std::string instance;
    std::string config_file;
    std::vector<CameraConfig> cameras;
    std::optional<std::uint32_t> slots;
    bool automatic = false;
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
            arguments.cameras.push_back(parse_camera_argument(*camera));
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
            // Added by the plugin manager for plugins with a device tree; cameras have none.
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
PluginConfig cameras_to_serve(const Arguments& arguments, const std::vector<Backend>& backends, spdlog::logger& logger)
{
    PluginConfig config;
    std::string source = "the command line";
    const std::string default_file = default_config_path();
    if (!arguments.cameras.empty())
    {
        config.cameras = arguments.cameras;
    }
    else if (!arguments.config_file.empty() || (!arguments.automatic && std::filesystem::exists(default_file)))
    {
        source = arguments.config_file.empty() ? default_file : arguments.config_file;
        config = load_config_file(source);
    }
    else
    {
        source = "the cameras found plugged in";
        config.cameras = discover_cameras(backends);
    }
    if (arguments.slots)
    {
        config.slot_count = *arguments.slots;
    }
    if (config.cameras.empty())
    {
        throw std::invalid_argument("there is no camera to serve in " + source + ". Name cameras in '" + default_file +
                                    "' or with --camera, or plug in a camera one of this build's backends can find");
    }
    logger.info("serving {} camera(s) from {}", config.cameras.size(), source);
    return config;
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
        // Every thread started below inherits this mask, so the signals wait for the loop here.
        sigset_t stop_signals;
        sigemptyset(&stop_signals);
        sigaddset(&stop_signals, SIGINT);
        sigaddset(&stop_signals, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);

        const PluginConfig config = cameras_to_serve(arguments, backends, *logger);
        std::string endpoint = arguments.endpoint;
        if (endpoint.empty())
        {
            endpoint = core::camera::default_endpoint_path(arguments.instance);
            core::camera::prepare_endpoint_directory(endpoint);
        }

        CameraHost host(endpoint);
        for (std::unique_ptr<CameraSource>& source : open_cameras(backends, config.cameras))
        {
            host.add_camera(std::move(source), config.slot_count);
        }
        host.start();
        logger->info("camera plugin serves {} camera(s) on '{}'", host.camera_count(), endpoint);

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
