// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Publishes camera_receiver's CUDA IPC streams to a LiveKit room as pre-encoded H.264 tracks.

#include "stream_worker.hpp"

#include <livekit_publisher/livekit_session.hpp>
#include <nvenc/cuda_context.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{

std::atomic<bool> g_stop{ false };

void on_signal(int)
{
    g_stop = true;
}

void usage()
{
    std::cerr
        << "usage: camera_livekit_bridge --livekit-url URL --livekit-token-file FILE --stream NAME=TRACK[:DIV] [--stream ...]\n"
           "         [--socket PATH] [--fps N] [--bitrate-mbps N] [--idr-period N] [--gpu N]\n         [--timestamp-basis monotonic_us|capture_monotonic_us|capture_unix_us]\n"
           "  NAME is a stream in camera_receiver's IPC catalog; TRACK is the LiveKit track name egocap expects;\n  DIV shrinks the picture on the GPU by that integer factor before encoding.\n";
}

std::string read_token(const std::string& path)
{
    std::ifstream file(path);
    std::string token;
    std::getline(file, token);
    while (!token.empty() && (token.back() == ' ' || token.back() == '\r' || token.back() == '\t'))
        token.pop_back();
    if (token.empty())
        throw std::runtime_error("cannot read a token from " + path);
    return token;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace camera_bridge;

    std::string url;
    std::string token_file;
    std::string socket_path = "/tmp/camera_receiver_avs1_cuda_ipc.sock";
    struct StreamSpec
    {
        std::string name;
        std::string track;
        uint32_t scale_div = 1;
    };
    std::vector<StreamSpec> streams;
    WorkerConfig defaults;
    int gpu = 0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto next = [&]() -> std::string
        {
            if (i + 1 >= argc)
            {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--livekit-url")
            url = next();
        else if (arg == "--livekit-token-file")
            token_file = next();
        else if (arg == "--socket")
            socket_path = next();
        else if (arg == "--fps")
            defaults.fps = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (arg == "--bitrate-mbps")
            defaults.bitrate_bps = static_cast<uint32_t>(std::atof(next().c_str()) * 1e6);
        else if (arg == "--idr-period")
            defaults.idr_period = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (arg == "--timestamp-basis")
        {
            const auto basis = parse_timestamp_basis(next());
            if (!basis)
            {
                std::cerr << "--timestamp-basis must be monotonic_us, capture_monotonic_us or capture_unix_us\n";
                return 2;
            }
            if (*basis == TimestampBasis::CaptureMonotonicUs)
            {
                std::cerr
                    << "--timestamp-basis capture_monotonic_us is not available: it needs a measured sender-to-host clock "
                       "mapping, which is not implemented. Use monotonic_us (frame-ready time) or capture_unix_us.\n";
                return 2;
            }
            defaults.timestamp_basis = *basis;
        }
        else if (arg == "--gpu")
            gpu = std::atoi(next().c_str());
        else if (arg == "--stream")
        {
            std::string spec = next();
            StreamSpec parsed;
            const auto colon = spec.rfind(':');
            if (colon != std::string::npos)
            {
                parsed.scale_div = static_cast<uint32_t>(std::atoi(spec.substr(colon + 1).c_str()));
                spec.erase(colon);
            }
            const auto eq = spec.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 == spec.size() || parsed.scale_div == 0)
            {
                std::cerr << "--stream needs NAME=TRACK or NAME=TRACK:DIV, got '" << spec << "'\n";
                return 2;
            }
            parsed.name = spec.substr(0, eq);
            parsed.track = spec.substr(eq + 1);
            streams.push_back(parsed);
        }
        else
        {
            usage();
            return 2;
        }
    }
    if (url.empty() || token_file.empty() || streams.empty() || defaults.fps == 0)
    {
        usage();
        return 2;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    try
    {
        CudaContext context(gpu);
        auto session = LiveKitSession::connect(url, read_token(token_file));
        std::vector<std::unique_ptr<StreamWorker>> workers;
        for (const auto& spec : streams)
        {
            WorkerConfig config = defaults;
            config.stream = spec.name;
            config.track = spec.track;
            config.scale_div = spec.scale_div;
            config.socket_path = socket_path;
            workers.push_back(std::make_unique<StreamWorker>(config, context, *session));
        }
        for (auto& worker : workers)
            worker->start();
        std::cerr << "[info] publishing " << workers.size() << " stream(s) on " << context.device_name() << "\n";

        auto last_report = std::chrono::steady_clock::now();
        std::vector<uint64_t> last_frames(workers.size(), 0);
        int exit_code = 0;
        while (!g_stop && exit_code == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (session->failed())
            {
                std::cerr << "[error] " << session->failure() << "\n";
                exit_code = 1;
            }
            for (const auto& worker : workers)
            {
                const std::string failure = worker->failure();
                if (!failure.empty())
                {
                    std::cerr << "[error] " << worker->config().stream << ": " << failure << "\n";
                    exit_code = 1;
                }
            }
            const auto now = std::chrono::steady_clock::now();
            if (now - last_report >= std::chrono::seconds(5))
            {
                const double seconds = std::chrono::duration<double>(now - last_report).count();
                for (std::size_t i = 0; i < workers.size(); ++i)
                {
                    const auto& stats = workers[i]->stats();
                    const uint64_t frames = stats.frames.load();
                    const double mean_ms =
                        frames == 0 ? 0.0 : static_cast<double>(stats.encode_us_total.load()) / frames / 1000.0;
                    std::cerr << "[stats] " << workers[i]->config().track << ": " << (frames - last_frames[i]) / seconds
                              << " fps, " << frames << " frames, " << stats.keyframes.load() << " keyframes, "
                              << stats.skipped.load() << " skipped, " << stats.rejected.load() << " rejected, encode "
                              << mean_ms << " ms mean, bitrate " << stats.bitrate_bps.load() / 1000
                              << " kbps (transport target " << stats.target_bps.load() / 1000 << " kbps, "
                              << stats.rate_changes.load() << " changes)";
                    const uint64_t ready_samples = stats.capture_to_ready_samples.load();
                    if (ready_samples > 0)
                    {
                        std::cerr << ", capture-to-ready "
                                  << static_cast<double>(stats.capture_to_ready_us_total.load()) / ready_samples / 1000.0
                                  << " ms mean " << stats.capture_to_ready_max_us.load() / 1000.0 << " ms max";
                    }
                    std::cerr << "\n";
                    last_frames[i] = frames;
                }
                last_report = now;
            }
        }
        for (auto& worker : workers)
            worker->stop();
        // Tracks unpublish when their workers are destroyed, which must happen before the session leaves the room.
        workers.clear();
        return exit_code;
    }
    catch (const std::exception& error)
    {
        std::cerr << "[error] " << error.what() << "\n";
        return 1;
    }
}
