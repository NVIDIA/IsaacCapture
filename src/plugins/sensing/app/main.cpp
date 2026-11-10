// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <sensing_control/control_handler.hpp>
#include <sensing_control/control_server.hpp>
#include <sensing_control/driver_log_capture.hpp>
#include <sensing_control/producer_supervisor.hpp>
#include <sensing_core/sensing_camera.hpp>
#include <sensing_core/sipl_camera.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
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
// Producer session and release
// =============================================================================

/// The release this binary was installed from (".../releases/<id>/..."), or a placeholder when run by hand.
static std::string release_id()
{
    char buffer[PATH_MAX];
    const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0)
        return "unknown";
    const std::string path(buffer, static_cast<size_t>(length));
    const std::string marker = "/releases/";
    const auto at = path.find(marker);
    if (at == std::string::npos)
        return "unmanaged";
    const auto start = at + marker.size();
    return path.substr(start, path.find('/', start) - start);
}

/// One capture-and-publish pipeline as the supervisor sees it.
class ProducerSession : public CameraSession
{
public:
    ProducerSession(const SensingConfig& config, const std::vector<StreamConfig>& streams)
        : m_camera(config, streams), m_last_stats(std::chrono::steady_clock::now())
    {
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "Capturing." << std::endl;
    }

    void update() override
    {
        m_camera.update();
        const auto now = std::chrono::steady_clock::now();
        if (now - m_last_stats >= std::chrono::seconds(5))
        {
            m_camera.print_stats();
            m_last_stats = now;
        }
    }

    void flush() override
    {
        m_camera.flush();
        m_camera.print_stats();
    }

    std::vector<StreamStatus> streams() const override
    {
        std::vector<StreamStatus> out;
        for (const auto& figures : m_camera.figures())
        {
            StreamStatus entry;
            entry.track = figures.track;
            entry.sensor_id = figures.sensor_id;
            entry.frames = figures.frames;
            entry.published = figures.published;
            out.push_back(std::move(entry));
        }
        return out;
    }

    /// Never called: ProducerActions refuses direct targets before the supervisor records one.
    void set_direct_target(const std::string&, const std::string&, uint16_t) override
    {
    }

private:
    SensingCamera m_camera;
    std::chrono::steady_clock::time_point m_last_stats;
};

/// This producer publishes to LiveKit and has no direct link to a PC, so the one action the channel can name is refused
/// with a reason instead of being accepted and ignored.
class ProducerActions : public ControlActions
{
public:
    ActionResult set_direct_target(const std::string&, const std::string&, uint16_t) override
    {
        return { false, "not_supported", "this producer publishes to LiveKit and has no direct link to set" };
    }
};

// =============================================================================
// Usage
// =============================================================================

static void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " --livekit-url=URL --livekit-token-file=PATH --add-stream=... [options]\n"
              << "\n"
              << "Publishes each SENSING sensor as a LiveKit camera track: SIPL capture, hardware H.264,\n"
              << "sent as it is.\n"
              << "\n"
              << "Streams (repeat for each sensor):\n"
              << "  --add-stream=sensor=N,livekit=NAME   SIPL pipeline index N, published as track NAME\n"
              << "  --list-sensors                       print the pipelines, then exit\n"
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
              << "Management channel (a WebSocket server; off unless a token file is given):\n"
              << "  --control-token-file=PATH file holding the shared secret clients must present\n"
              << "  --control-port=N          port to listen on (default: 5610)\n"
              << "  --control-bind=ADDR       IPv4 address to listen on (default: 0.0.0.0)\n"
              << "  --retry-seconds=N         wait between attempts to start the cameras (default: 5)\n"
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
    ControlServerConfig control_config;
    std::string control_token_file;
    uint32_t retry_seconds = 5;
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
        else if (arg.find("--control-token-file=") == 0)
        {
            control_token_file = arg.substr(21);
        }
        else if (arg.find("--control-port=") == 0)
        {
            const unsigned long port = std::stoul(arg.substr(15));
            if (port < 1 || port > 65535)
                throw std::runtime_error("--control-port must be from 1 to 65535");
            control_config.port = static_cast<uint16_t>(port);
        }
        else if (arg.find("--control-bind=") == 0)
        {
            control_config.bind_address = arg.substr(15);
        }
        else if (arg.find("--retry-seconds=") == 0)
        {
            retry_seconds = static_cast<uint32_t>(std::stoul(arg.substr(16)));
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

    // Query is hardware-free -- it only parses the driver database and the JSON -- so --list-sensors
    // works on a rig whose cameras are unplugged, and the poll interval below can be derived before
    // anything is opened.
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
    std::cout << "SENSING Camera Plugin Starting (SIPL)" << std::endl;
    std::cout << "============================================================" << std::endl;

    // A stream naming a sensor that does not exist is a configuration error, not a fault to retry.
    for (const auto& stream : stream_configs)
    {
        const bool known =
            std::any_of(sensors.begin(), sensors.end(), [&](const SensorInfo& s) { return s.id == stream.sensor_id; });
        if (!known)
            throw std::runtime_error("sensor=" + std::to_string(stream.sensor_id) + " is not a pipeline of '" +
                                     camera_config.platform_config_name + "'; run --list-sensors");
    }

    // SiplCamera::latest() is a non-blocking mailbox read, so poll at roughly twice the fastest sensor's frame rate
    // rather than spinning a core flat out. Capture itself runs on SIPL's own threads and is unaffected.
    double fastest_fps = 30.0;
    for (const auto& s : sensors)
        fastest_fps = std::max(fastest_fps, s.fps);

    ProducerStatus status(release_id());
    std::vector<StreamStatus> configured;
    for (const auto& stream : stream_configs)
    {
        StreamStatus entry;
        entry.track = stream.livekit_track;
        entry.sensor_id = stream.sensor_id;
        configured.push_back(entry);
    }

    // The cameras and the room are opened by the factory, again after any failure, so an unplugged camera is a state
    // to report and retry, not the end of the process.
    const SessionFactory factory = [&](const DirectTargets&) -> std::unique_ptr<CameraSession>
    {
        // The camera driver explains a failure only in its own log lines, so they are collected while the cameras
        // start and attached to the failure; the fault table reads them (an open circuit, a lock timeout, an I2C NACK).
        DriverLogCapture driver_log;
        try
        {
            auto session = std::make_unique<ProducerSession>(camera_config, stream_configs);
            driver_log.finish();
            return session;
        }
        catch (const std::exception& e)
        {
            const std::string lines = driver_log.finish();
            throw std::runtime_error(lines.empty() ? std::string(e.what()) :
                                                     std::string(e.what()) + " | driver log: " + lines);
        }
    };

    // Destruction runs in reverse declaration order, so the server (declared last) is stopped and joined while the
    // handler, supervisor and status its thread calls into still exist, on return and on exception alike.
    // `channel` is what the supervisor publishes through; it is set once the server exists.
    ControlServer* channel = nullptr;
    const auto publish = [&channel](const std::string& text)
    {
        if (channel != nullptr)
            channel->broadcast(text);
    };
    SupervisorConfig supervisor_config;
    supervisor_config.retry_interval = std::chrono::seconds(std::max<uint32_t>(retry_seconds, 1));
    supervisor_config.poll_interval = std::chrono::milliseconds(static_cast<int64_t>(500.0 / fastest_fps));
    ProducerSupervisor supervisor(status, configured, factory, publish, supervisor_config);
    ProducerActions actions;
    ControlHandler handler(status, actions);
    std::unique_ptr<ControlServer> server;

    if (control_token_file.empty())
    {
        std::cout << "Management channel: off (give --control-token-file to turn it on)" << std::endl;
    }
    else
    {
        control_config.token = read_secret_file(control_token_file);
        server = std::make_unique<ControlServer>(
            control_config, [&handler](const std::string& text) { return handler.handle(text); });
        channel = server.get();
        std::cout << "Management channel: listening on " << control_config.bind_address << ":" << server->port()
                  << std::endl;
    }

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Running. Press Ctrl+C to stop." << std::endl;

    supervisor.run([] { return g_stop_requested.load(std::memory_order_relaxed); });
    channel = nullptr; // nothing publishes once the run has ended

    std::cout << "------------------------------------------------------------" << std::endl;
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
