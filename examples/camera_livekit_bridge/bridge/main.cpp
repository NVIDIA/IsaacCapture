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
        << "usage: camera_livekit_bridge --livekit-url URL --livekit-token-file FILE --stream NAME=TRACK [--stream ...]\n"
           "         [--socket PATH] [--fps N] [--bitrate-mbps N] [--idr-period N] [--gpu N]\n"
           "  NAME is a stream in camera_receiver's IPC catalog; TRACK is the LiveKit track name egocap expects.\n";
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
    std::vector<std::pair<std::string, std::string>> streams;
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
        else if (arg == "--gpu")
            gpu = std::atoi(next().c_str());
        else if (arg == "--stream")
        {
            const std::string spec = next();
            const auto eq = spec.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 == spec.size())
            {
                std::cerr << "--stream needs NAME=TRACK, got '" << spec << "'\n";
                return 2;
            }
            streams.emplace_back(spec.substr(0, eq), spec.substr(eq + 1));
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
        for (const auto& [name, track] : streams)
        {
            WorkerConfig config = defaults;
            config.stream = name;
            config.track = track;
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
                              << mean_ms << " ms mean\n";
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
