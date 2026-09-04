// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "core/frame_sink.hpp"
#include "core/sensing_camera.hpp"
#include "core/sipl_camera.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

using namespace plugins::sensing;

// =============================================================================
// Signal handling
// =============================================================================

static std::atomic<bool> g_stop_requested{ false };

void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
    {
        g_stop_requested.store(true, std::memory_order_relaxed);
    }
}

// =============================================================================
// Argument parsers
// =============================================================================

static StreamConfig parse_stream_arg(const std::string& arg)
{
    StreamConfig cfg{};
    bool has_sensor = false;

    std::istringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        auto eq = token.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("Invalid key=value in --add-stream: '" + token + "'");

        auto key = token.substr(0, eq);
        auto val = token.substr(eq + 1);

        if (key == "sensor")
        {
            cfg.sensor_id = static_cast<uint32_t>(std::stoul(val));
            has_sensor = true;
        }
        else if (key == "output")
        {
            cfg.output_path = val;
        }
        else if (key == "ipc")
        {
            cfg.ipc_socket_path = val;
        }
        else
        {
            throw std::runtime_error("Unknown key in --add-stream: '" + key + "'");
        }
    }

    if (!has_sensor)
        throw std::runtime_error("--add-stream requires sensor=<sipl-pipeline-index>");
    if (cfg.output_path.empty() && cfg.ipc_socket_path.empty())
        throw std::runtime_error("--add-stream requires output=<path> or ipc=<socket>, or both");

    return cfg;
}

/// "0x0000 0x1100" -> {0x0000, 0x1100}. Same spelling as nvsipl_camera -m, so a
/// working vendor command line can be copied across verbatim.
static std::vector<uint32_t> parse_link_masks(const std::string& arg)
{
    std::vector<uint32_t> masks;
    std::istringstream ss(arg);
    std::string token;
    while (ss >> token)
    {
        masks.push_back(static_cast<uint32_t>(std::stoul(token, nullptr, 0)));
    }
    if (masks.empty())
        throw std::runtime_error("--link-masks needs at least one mask, e.g. --link-masks='0x0000 0x1100'");
    return masks;
}

/// The vendored platform config, resolved next to the executable so it works
/// from the build tree and the install tree alike, whatever the cwd. Empty when
/// it is not there, which leaves --platform-config required.
static std::string default_platform_config()
{
    char buf[PATH_MAX];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0)
        return {};
    buf[n] = '\0';
    const std::string exe(buf);
    const auto slash = exe.rfind('/');
    if (slash == std::string::npos)
        return {};
    const std::string candidate = exe.substr(0, slash) + "/configs/shw5g.json";
    return ::access(candidate.c_str(), R_OK) == 0 ? candidate : std::string{};
}

// =============================================================================
// Usage
// =============================================================================

void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [options] --add-stream ...\n"
              << "\nStream Configuration (repeatable):\n"
              << "  --add-stream sensor=<id>[,output=<path>][,ipc=<socket>]\n"
              << "      sensor: SIPL pipeline index, as --list-sensors reports it. This is NOT\n"
              << "              the GMSL link index or the JSON sensorInfo.id.\n"
              << "      output: file path for this stream's H.264 data\n"
              << "      ipc:    Unix socket serving raw RGBA8 frames as CUDA memory to\n"
              << "              another process (camera_viz `type: cuda_ipc`). No encode.\n"
              << "      At least one of output/ipc is required; both may be given.\n"
              << "\nPlatform (resolution and frame rate come from here, not from flags):\n"
              << "  --platform-config=PATH  platform JSON (default: configs/shw5g.json beside this\n"
              << "                          binary; point it at the vendor package to test a\n"
              << "                          newer driver drop)\n"
              << "  --config-name=NAME      named config inside it (default: SHW5G_2)\n"
              << "  --link-masks='M M'      one mask per deserializer (default: '0x0000 0x1100')\n"
              << "  --nito-dir=DIR          ISP tuning directory (default: /var/nvidia/nvcam/settings/sipl)\n"
              << "  --list-sensors          print what the platform config resolves to, then exit\n"
              << "\nEncoding:\n"
              << "  --bitrate=N         H.264 average bitrate in bps (default: 40000000)\n"
              << "  --peak-bitrate=N    VBR ceiling in bps; 0 selects CBR (default: 60000000)\n"
              << "  --gop=N             IDR period in frames (default: fps*5)\n"
              << "\nGeneral:\n"
              << "  --gpu-id=N          CUDA device index (default: 0)\n"
              << "  --isp0-buffers=N    ISP0 buffers per sensor (default: 6)\n"
              << "  --full-range        Treat luma as full range instead of broadcast range\n"
              << "  --swap-uv           Swap the chroma planes\n"
              << "  --mcap-filename=PATH  Record per-frame metadata to an MCAP file\n"
              << "  --help              Show this help message\n"
              << "\nExamples:\n"
              << "  " << program_name << " --list-sensors\n"
              << "  " << program_name << " --add-stream=sensor=0,ipc=/tmp/sensing0.sock \\\n"
              << "      --add-stream=sensor=1,ipc=/tmp/sensing1.sock\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
try
{
    SensingConfig camera_config;
    std::map<uint32_t, StreamConfig> stream_map;
    std::string mcap_filename;
    bool list_sensors = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg.find("--add-stream=") == 0)
        {
            auto cfg = parse_stream_arg(arg.substr(13));
            // Overwriting would silently drop the earlier stream's output= or
            // ipc=; both belong in one --add-stream for a given sensor.
            if (!stream_map.emplace(cfg.sensor_id, cfg).second)
            {
                throw std::runtime_error("sensor " + std::to_string(cfg.sensor_id) +
                                         " given twice; put output= and ipc= in a single --add-stream");
            }
        }
        else if (arg.find("--platform-config=") == 0)
        {
            camera_config.platform_config_json = arg.substr(18);
        }
        else if (arg.find("--config-name=") == 0)
        {
            camera_config.platform_config_name = arg.substr(14);
        }
        else if (arg.find("--link-masks=") == 0)
        {
            camera_config.link_masks = parse_link_masks(arg.substr(13));
        }
        else if (arg.find("--nito-dir=") == 0)
        {
            camera_config.nito_dir = arg.substr(11);
        }
        else if (arg == "--list-sensors")
        {
            list_sensors = true;
        }
        else if (arg.find("--bitrate=") == 0)
        {
            camera_config.bitrate_bps = static_cast<uint32_t>(std::stoul(arg.substr(10)));
        }
        else if (arg.find("--peak-bitrate=") == 0)
        {
            camera_config.peak_bitrate_bps = static_cast<uint32_t>(std::stoul(arg.substr(15)));
        }
        else if (arg.find("--gop=") == 0)
        {
            camera_config.gop = static_cast<uint32_t>(std::stoul(arg.substr(6)));
        }
        else if (arg.find("--gpu-id=") == 0)
        {
            camera_config.gpu_id = std::stoi(arg.substr(9));
        }
        else if (arg.find("--isp0-buffers=") == 0)
        {
            camera_config.isp0_buffers = static_cast<uint32_t>(std::stoul(arg.substr(15)));
        }
        else if (arg == "--full-range")
        {
            camera_config.full_range = true;
        }
        else if (arg == "--swap-uv")
        {
            camera_config.swap_uv = true;
        }
        else if (arg.find("--mcap-filename=") == 0)
        {
            mcap_filename = arg.substr(16);
        }
        else if (arg.find("--plugin-root-id=") == 0)
        {
            // plugin-root-id is a default argument, so we don't need to store it
        }
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (camera_config.platform_config_json.empty())
    {
        camera_config.platform_config_json = default_platform_config();
    }
    if (camera_config.platform_config_json.empty())
    {
        std::cerr << "Error: no platform config. configs/shw5g.json was not found beside this\n"
                     "binary, so pass --platform-config=PATH. SIPL cannot enumerate sensors\n"
                     "without one."
                  << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Query is hardware-free -- it only parses the driver database and the JSON
    // -- so --list-sensors works on a rig whose cameras are unplugged, and the
    // poll interval below can be derived before anything is opened.
    const auto sensors = SiplCamera::query(
        camera_config.platform_config_json, camera_config.platform_config_name, camera_config.link_masks);

    if (list_sensors)
    {
        std::cout << camera_config.platform_config_name << ": " << sensors.size() << " sensor(s)\n";
        for (const auto& s : sensors)
        {
            std::cout << "  sensor=" << s.id << "  " << s.name << "  " << s.width << "x" << s.height << " @ " << s.fps
                      << " fps\n";
        }
        return 0;
    }

    if (stream_map.empty())
    {
        std::cerr << "Error: at least one --add-stream is required." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::vector<StreamConfig> stream_configs;
    stream_configs.reserve(stream_map.size());
    for (auto& [_, cfg] : stream_map)
    {
        stream_configs.push_back(std::move(cfg));
    }

    // This process captures and never renders, but EGL is on its critical path:
    // NvBufSurfaceMapEglImage() is how a SIPL NvSciBufObj reaches CUDA, and
    // libnvbufsurface resolves its own display via
    // eglGetDisplay(EGL_DEFAULT_DISPLAY). GLVND hands out Mesa's EGL rather than
    // Tegra's whenever DISPLAY names an X server Tegra EGL cannot drive (Xvfb,
    // X11 forwarding), and that choice cannot be made per-call -- DISPLAY has to
    // be gone from the environment before the first EGL call.
    ::unsetenv("DISPLAY");

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "============================================================" << std::endl;
    std::cout << "SENSING Camera Plugin Starting (SIPL)" << std::endl;
    std::cout << "============================================================" << std::endl;

    SensingCamera camera(camera_config, stream_configs, create_frame_sink(stream_configs, mcap_filename));

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Running capture loop. Press Ctrl+C to stop." << std::endl;

    constexpr auto stats_interval = std::chrono::seconds(5);
    auto last_stats_time = std::chrono::steady_clock::now();

    // SiplCamera::latest() is a non-blocking mailbox read, so poll at roughly
    // twice the fastest sensor's frame rate rather than spinning a core flat
    // out. Capture itself runs on SIPL's own threads and is unaffected.
    double fastest_fps = 30.0;
    for (const auto& s : sensors)
    {
        fastest_fps = std::max(fastest_fps, s.fps);
    }
    const auto poll_interval = std::chrono::microseconds(static_cast<int64_t>(500'000.0 / fastest_fps));

    while (!g_stop_requested.load(std::memory_order_relaxed))
    {
        camera.update();

        auto now = std::chrono::steady_clock::now();
        if (now - last_stats_time >= stats_interval)
        {
            camera.print_stats();
            last_stats_time = now;
        }

        std::this_thread::sleep_for(poll_interval);
    }

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Shutting down SENSING Camera Plugin..." << std::endl;
    camera.flush();
    camera.print_stats();
    std::cout << "Plugin stopped" << std::endl;
    std::cout << "============================================================" << std::endl;

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
