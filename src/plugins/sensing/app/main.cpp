// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <sensing_core/camera_backend.hpp>
#include <sensing_core/capture_factory.hpp>
#include <sensing_core/sensing_camera.hpp>

#ifdef SENSING_HAS_SIPL
#    include <sensing_core/sipl_camera.hpp>
#endif
#ifdef SENSING_HAS_ARGUS
#    include <sensing_core/argus_capture.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdlib>
#include <fstream>
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

/// "sensor=N,livekit=NAME" -> one sensor and the track that carries it.
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
        else if (key == "livekit")
        {
            cfg.livekit_track = val;
        }
        else
        {
            throw std::runtime_error("Unknown key in --add-stream: '" + key + "'");
        }
    }

    if (!has_sensor || cfg.livekit_track.empty())
        throw std::runtime_error("--add-stream requires sensor=<sipl-pipeline-index>,livekit=<track>");

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

static void print_usage(const char* program_name)
{
    std::cout
        << "Usage: " << program_name << " --livekit-url=URL --livekit-token-file=PATH --add-stream=... [options]\n"
        << "\n"
        << "Publishes each SENSING sensor as a LiveKit camera track: SIPL capture, hardware H.264,\n"
        << "sent as it is.\n"
        << "\n"
        << "Streams (repeat for each sensor):\n"
        << "  --add-stream=sensor=N,livekit=NAME   camera N (SIPL pipeline or Argus device index), published as track NAME\n"
        << "  --list-sensors                       print the cameras, then exit\n"
        << "  --camera-backend=auto|sipl|argus     auto picks the one backend that matches (default);\n"
        << "                                       (a backend must also be built in)\n"
        << "  --argus-mode=N                       Argus sensor mode index (default: 0)\n"
        << "\n"
        << "Platform:\n"
        << "  --platform-config=PATH  platform JSON (default: configs/shw5g.json beside this\n"
        << "                          binary)\n"
        << "  --config-name=NAME      named config inside it (default: SHW5G_2)\n"
        << "  --link-masks='M M'      one mask per deserializer (default: '0x0000 0x1100')\n"
        << "  --nito-dir=DIR          ISP tuning directory (default: /var/nvidia/nvcam/settings/sipl)\n"
        << "\n"
        << "LiveKit:\n"
        << "  --livekit-url=URL         server, e.g. wss://livekit.example.org\n"
        << "  --livekit-token-file=PATH file holding the join token; it names the room and the\n"
        << "                            participant identity\n"
        << "  --livekit-fps=N           publication rate cap (default: 30)\n"
        << "  --livekit-scale-div=N     reduce every track by this integer divisor (default: 1)\n"
        << "  --livekit-bitrate=N       target bitrate of each track's encoder in bps (default: 8000000)\n"
        << "  --gop=N                   IDR period in frames (default: fps*5)\n"
        << "\n"
        << "Other:\n"
        << "  --gpu-id=N          CUDA device index (default: 0)\n"
        << "  --isp0-buffers=N    ISP0 buffers per sensor (default: 6)\n"
        << "  --full-range        full-range YUV\n"
        << "  --swap-uv           swap the chroma planes\n"
        << "\n"
        << "Example:\n"
        << "  " << program_name << " --livekit-url=wss://livekit.example.org \\\n"
        << "      --livekit-token-file=join.token --add-stream=sensor=0,livekit=sensing-left \\\n"
        << "      --add-stream=sensor=1,livekit=sensing-right\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
try
{
    SensingConfig camera_config;
    std::map<uint32_t, StreamConfig> stream_map;
    std::string livekit_token_file;
    bool list_sensors = false;
    CameraBackend requested_backend = CameraBackend::Auto;

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
            if (!stream_map.emplace(cfg.sensor_id, cfg).second)
                throw std::runtime_error("sensor " + std::to_string(cfg.sensor_id) + " given twice");
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
        else if (arg.find("--argus-mode=") == 0)
        {
            camera_config.argus_sensor_mode = static_cast<uint32_t>(std::stoul(arg.substr(13)));
        }
        else if (arg.find("--camera-backend=") == 0)
        {
            const auto parsed = parse_camera_backend(arg.substr(17));
            if (!parsed)
            {
                std::cerr << "Error: --camera-backend must be auto, argus or sipl." << std::endl;
                return 1;
            }
            requested_backend = *parsed;
        }
        else if (arg.find("--livekit-url=") == 0)
        {
            camera_config.livekit.url = arg.substr(14);
        }
        else if (arg.find("--livekit-token-file=") == 0)
        {
            livekit_token_file = arg.substr(21);
        }
        else if (arg.find("--livekit-fps=") == 0)
        {
            camera_config.livekit.fps = static_cast<uint32_t>(std::stoul(arg.substr(14)));
        }
        else if (arg.find("--livekit-bitrate=") == 0)
        {
            camera_config.livekit.bitrate_bps = static_cast<uint32_t>(std::stoul(arg.substr(18)));
        }
        else if (arg.find("--livekit-scale-div=") == 0)
        {
            camera_config.livekit.scale_div = static_cast<uint32_t>(std::stoul(arg.substr(20)));
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

    // Each backend reports whether it can open the configured rig; probing parses configuration or enumerates
    // devices and opens no camera, so --list-sensors works on a rig whose cameras are unplugged.
    std::vector<uint32_t> camera_ids;
    for (const auto& entry : stream_map)
        camera_ids.push_back(entry.first);

    std::vector<CameraInfo> sipl_cameras;
    std::vector<CameraInfo> argus_cameras;
    BackendProbe sipl_probe;
    sipl_probe.backend = CameraBackend::Sipl;
    BackendProbe argus_probe;
    argus_probe.backend = CameraBackend::Argus;

#ifdef SENSING_HAS_SIPL
    sipl_probe.available = true;
    // An explicit --camera-backend names the one backend to use: the other SDK is not initialised just to be probed.
    if (requested_backend != CameraBackend::Argus)
    {
        if (camera_config.platform_config_json.empty())
            camera_config.platform_config_json = default_platform_config();
        if (camera_config.platform_config_json.empty())
        {
            sipl_probe.detail =
                "no platform config: configs/shw5g.json was not found beside this binary; pass "
                "--platform-config=PATH";
        }
        else
        {
            // The query parses the configuration; it does not prove that the cameras are present.
            try
            {
                for (const auto& s : SiplCamera::query(camera_config.platform_config_json,
                                                       camera_config.platform_config_name, camera_config.link_masks))
                    sipl_cameras.push_back({ s.id, s.name, s.width, s.height, s.fps });
                sipl_probe.matches = !sipl_cameras.empty();
                if (!sipl_probe.matches)
                    sipl_probe.detail = "the platform configuration names no sensors";
            }
            catch (const std::exception& e)
            {
                sipl_probe.detail = e.what();
            }
        }
    }
#else
    sipl_probe.detail = "this build has no SIPL support";
#endif

#ifdef SENSING_HAS_ARGUS
    argus_probe.available = true;
    if (requested_backend != CameraBackend::Sipl)
    {
        try
        {
            argus_cameras = ArgusCapture::query(camera_config.argus_sensor_mode);
            // Every requested camera must exist; with none requested (--list-sensors) any camera counts.
            argus_probe.matches = !argus_cameras.empty();
            for (uint32_t id : camera_ids)
            {
                const bool found = std::any_of(
                    argus_cameras.begin(), argus_cameras.end(), [id](const CameraInfo& c) { return c.id == id; });
                if (!found)
                {
                    argus_probe.matches = false;
                    argus_probe.detail = "no Argus camera device " + std::to_string(id);
                }
            }
            if (argus_cameras.empty())
                argus_probe.detail = "Argus reports no camera devices";
        }
        catch (const std::exception& e)
        {
            argus_probe.detail = e.what();
        }
    }
#else
    argus_probe.detail = "this build has no Argus support (configure with -DSENSING_WITH_ARGUS=ON)";
#endif

    const CameraBackend backend = select_camera_backend(requested_backend, { sipl_probe, argus_probe });
    const std::vector<CameraInfo>& cameras = backend == CameraBackend::Argus ? argus_cameras : sipl_cameras;
    std::cout << "Camera backend: " << camera_backend_name(backend) << std::endl;

    if (list_sensors)
    {
        std::cout << cameras.size() << " camera(s)\n";
        for (const auto& c : cameras)
        {
            std::cout << "  sensor=" << c.id << "  " << c.name << "  " << c.width << "x" << c.height << " @ " << c.fps
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

    // Check the settings that need no hardware or network before anything is opened.
    if (camera_config.livekit.url.empty() || livekit_token_file.empty())
        throw std::runtime_error("--livekit-url and --livekit-token-file are required");
    if (camera_config.livekit.fps == 0)
        throw std::runtime_error("--livekit-fps must be at least 1");
    if (camera_config.livekit.bitrate_bps < 100'000)
        throw std::runtime_error("--livekit-bitrate must be at least 100000");

    // The token is read from a file so it never appears in a process listing, and is not printed.
    std::ifstream token_file(livekit_token_file);
    if (!token_file)
        throw std::runtime_error("cannot read --livekit-token-file " + livekit_token_file);
    std::getline(token_file, camera_config.livekit.token);
    const auto last = camera_config.livekit.token.find_last_not_of(" \t\r\n");
    camera_config.livekit.token.erase(last == std::string::npos ? 0 : last + 1);
    if (camera_config.livekit.token.empty())
        throw std::runtime_error("--livekit-token-file " + livekit_token_file + " is empty");

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
    std::cout << "SENSING Camera Plugin Starting (" << camera_backend_name(backend) << ")" << std::endl;
    std::cout << "============================================================" << std::endl;

    SensingCamera camera(camera_config, make_capture(backend, camera_config, camera_ids), stream_configs);

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Running capture loop. Press Ctrl+C to stop." << std::endl;

    constexpr auto stats_interval = std::chrono::seconds(5);
    auto last_stats_time = std::chrono::steady_clock::now();

    // SiplCamera::latest() is a non-blocking mailbox read, so poll at roughly
    // twice the fastest sensor's frame rate rather than spinning a core flat
    // out. Capture itself runs on SIPL's own threads and is unaffected.
    double fastest_fps = 30.0;
    for (const auto& c : cameras)
    {
        fastest_fps = std::max(fastest_fps, c.fps);
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
