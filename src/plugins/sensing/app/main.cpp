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

/// "sensor=N,name=NAME[,direct=HOST:PORT]" -> one sensor and the stream that carries it.
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
        else if (key == "name")
        {
            // What the PC, the management channel and the logs call this stream.
            cfg.name = val;
        }
        else if (key == "direct")
        {
            // HOST:PORT of this stream's PC video-stage port; the producer connects and reconnects.
            const auto colon = val.rfind(':');
            if (colon == std::string::npos || colon == 0 || colon + 1 == val.size())
                throw std::runtime_error("--add-stream direct= must be HOST:PORT");
            cfg.direct_host = val.substr(0, colon);
            cfg.direct_port = static_cast<uint16_t>(std::stoul(val.substr(colon + 1)));
        }
        else
        {
            throw std::runtime_error("Unknown key in --add-stream: '" + key + "'");
        }
    }

    if (!has_sensor || cfg.name.empty())
        throw std::runtime_error("--add-stream requires sensor=<camera-index>,name=<stream name>");

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
        << "Usage: " << program_name << " --add-stream=... [options]\n"
        << "\n"
        << "Captures each SENSING sensor (SIPL or Argus), encodes it as H.264 on the hardware encoder and sends\n"
        << "it over TCP to the PC video stage, which publishes it onwards.\n"
        << "\n"
        << "Streams (repeat for each sensor):\n"
        << "  --add-stream=sensor=N,name=NAME[,direct=HOST:PORT]\n"
        << "                                       camera N (SIPL or Argus device), called NAME;\n"
        << "                                       direct= is where the PC video stage listens for it\n"
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
        << "Encoding:\n"
        << "  --fps=N                   frame rate cap of each stream (default: 30)\n"
        << "  --scale-div=N             reduce every stream by this integer divisor (default: 1)\n"
        << "  --timestamp-basis=B       what each frame's timestamp is: monotonic_us (default) is this host's\n"
        << "                            monotonic clock when the frame was ready to encode; capture_monotonic_us\n"
        << "                            (UNVERIFIED on hardware) is the sensor's capture event on that clock and fails\n"
        << "                            the stream if it cannot be placed. The direct link carries this clock only.\n"
        << "                            The streams file on the PC must declare the same basis.\n"
        << "  --frame-correlation=C     how an encoded unit is matched to its capture: stamp (default) by the stamp the\n"
        << "                            encoder returns, sending a unit that matches none without a capture time; fifo\n"
        << "                            assumes one unit per input in order (approximate; not for recording)\n"
        << "  --bitrate=N               target bitrate of each stream's encoder in bps (default: 8000000)\n"
        << "  --gop=N                   IDR period in frames (default: fps*5)\n"
        << "\n"
        << "Other:\n"
        << "  --gpu-id=N          CUDA device index (default: 0)\n"
        << "  --isp0-buffers=N    ISP0 buffers per sensor (default: 6)\n"
        << "  --full-range        full-range YUV\n"
        << "  --swap-uv           swap the chroma planes\n"
        << "\n"
        << "Example:\n"
        << "  " << program_name << " --add-stream=sensor=0,name=sensing-left,direct=192.168.1.205:5600 \\\n"
        << "      --add-stream=sensor=1,name=sensing-right,direct=192.168.1.205:5601\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
try
{
    SensingConfig camera_config;
    std::map<uint32_t, StreamConfig> stream_map;
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
        else if (arg.find("--fps=") == 0)
        {
            camera_config.encode.fps = static_cast<uint32_t>(std::stoul(arg.substr(6)));
        }
        else if (arg.find("--timestamp-basis=") == 0)
        {
            const std::string basis = arg.substr(18);
            if (basis == "monotonic_us")
                camera_config.encode.timestamp_basis = TimestampBasis::FrameReadyMonotonic;
            else if (basis == "capture_monotonic_us")
                camera_config.encode.timestamp_basis = TimestampBasis::SensorCaptureMonotonic;
            else
            {
                std::cerr
                    << "Error: --timestamp-basis must be monotonic_us or capture_monotonic_us: the direct link carries the sender's monotonic clock only."
                    << std::endl;
                return 1;
            }
        }
        else if (arg.find("--frame-correlation=") == 0)
        {
            const std::string how = arg.substr(20);
            if (how == "stamp")
                camera_config.encode.correlation = FrameCorrelation::Stamp;
            else if (how == "fifo")
                camera_config.encode.correlation = FrameCorrelation::Fifo;
            else
            {
                std::cerr << "Error: --frame-correlation must be stamp or fifo." << std::endl;
                return 1;
            }
        }
        else if (arg.find("--bitrate=") == 0)
        {
            camera_config.encode.bitrate_bps = static_cast<uint32_t>(std::stoul(arg.substr(10)));
        }
        else if (arg.find("--scale-div=") == 0)
        {
            camera_config.encode.scale_div = static_cast<uint32_t>(std::stoul(arg.substr(12)));
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

    // This process captures and never renders, but EGL is on its critical path:
    // NvBufSurfaceMapEglImage() is how a SIPL NvSciBufObj reaches CUDA, and
    // libnvbufsurface resolves its own display via
    // eglGetDisplay(EGL_DEFAULT_DISPLAY). GLVND hands out Mesa's EGL rather than
    // Tegra's whenever DISPLAY names an X server Tegra EGL cannot drive (Xvfb,
    // X11 forwarding), and that choice cannot be made per-call -- DISPLAY has to
    // be gone from the environment before the first EGL call, including the probes below.
    ::unsetenv("DISPLAY");

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
            match_requested_cameras(argus_cameras, camera_ids, "Argus", argus_probe);
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
    if (camera_config.encode.fps == 0)
        throw std::runtime_error("--fps must be at least 1");
    if (camera_config.encode.bitrate_bps < 100'000)
        throw std::runtime_error("--bitrate must be at least 100000");
    {
        const bool any_direct = std::any_of(
            stream_configs.begin(), stream_configs.end(), [](const StreamConfig& s) { return !s.direct_host.empty(); });
        if (!any_direct)
            std::cerr << "Warning: no stream has a direct=HOST:PORT destination: nothing will receive the video. "
                         "Give --add-stream a direct=HOST:PORT."
                      << std::endl;
    }

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

    // acquire_latest() is non-blocking, so poll at roughly twice the fastest sensor's frame rate rather than
    // spinning a core flat out. SIPL captures on its own threads and is unaffected; with Argus this loop also
    // returns buffers to the camera, so a stalled loop stalls capture.
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
