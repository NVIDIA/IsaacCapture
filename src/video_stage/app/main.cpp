// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "livekit_output.hpp"
#include "stream_worker.hpp"

#include <video_stage/cuda_context.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace video_stage;

namespace
{

std::atomic<bool> g_stop{ false };

void on_signal(int)
{
    g_stop = true;
}

void usage(const char* program)
{
    std::cout
        << "Usage: " << program << " --stream=PORT:TRACK ... --livekit-url=URL --livekit-token-file=PATH\n"
        << "       " << program << " --stream=PORT:file=PATH.h264 ...\n"
        << "       [--operation=none|sine-warp] [--amplitude=PX] [--bitrate=BPS] [--idr-period=N] [--max-upstream-uncertainty-ms=MS] [--stats-interval=S] [--once]\n"
        << "One --stream per camera. Each listens on PORT for the Orin's direct connection (AVS1 H.264 records plus a\n"
        << "control link) and publishes the access units as LiveKit track TRACK (all tracks under one participant) or\n"
        << "writes them to a file. By default (--operation=none) the Orin's access units are forwarded untouched, with no\n"
        << "decode, encode or GPU. --operation=sine-warp decodes on NVDEC, applies the example sine warp and re-encodes on\n"
        << "NVENC, as a test of the GPU stage. Serves one Orin\n"
        << "session after another unless --once. Prints per-stage percentiles at the end.\n";
}

std::string read_token(const std::string& path)
{
    std::ifstream in(path);
    std::string token;
    if (!in || !std::getline(in, token) || token.empty())
        throw std::runtime_error("cannot read a token from " + path);
    return token;
}

// "PORT:TRACK" or "PORT:file=PATH".
StreamSpec parse_stream(const std::string& text)
{
    const auto colon = text.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == text.size())
        throw std::runtime_error("--stream must be PORT:TRACK or PORT:file=PATH, got '" + text + "'");
    StreamSpec spec;
    spec.port = static_cast<uint16_t>(std::stoul(text.substr(0, colon)));
    const std::string rest = text.substr(colon + 1);
    if (rest.rfind("file=", 0) == 0)
        spec.file = rest.substr(5);
    else
        spec.track = rest;
    return spec;
}

} // namespace

int main(int argc, char** argv)
try
{
    std::vector<StreamSpec> specs;
    std::string livekit_url;
    std::string token_file;
    StreamSettings settings;
    int stats_interval_s = 5;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.rfind("--stream=", 0) == 0)
            specs.push_back(parse_stream(arg.substr(9)));
        else if (arg.rfind("--livekit-url=", 0) == 0)
            livekit_url = arg.substr(14);
        else if (arg.rfind("--livekit-token-file=", 0) == 0)
            token_file = arg.substr(21);
        else if (arg.rfind("--stats-interval=", 0) == 0)
            stats_interval_s = std::stoi(arg.substr(17));
        else if (arg == "--once")
            settings.once = true;
        else if (arg.rfind("--operation=", 0) == 0)
        {
            const std::string name = arg.substr(12);
            if (name == "none")
                settings.operation = Operation::None;
            else if (name == "sine-warp")
                settings.operation = Operation::SineWarp;
            else
            {
                std::cerr << "--operation must be none or sine-warp\n";
                return 1;
            }
        }
        else if (arg.rfind("--amplitude=", 0) == 0)
            settings.warp.amplitude_px = std::stof(arg.substr(12));
        else if (arg.rfind("--bitrate=", 0) == 0)
        {
            settings.pipeline.bitrate_bps = static_cast<uint32_t>(std::stoul(arg.substr(10)));
            settings.livekit.max_bitrate_bps = settings.pipeline.bitrate_bps;
        }
        else if (arg.rfind("--idr-period=", 0) == 0)
            settings.pipeline.idr_period = static_cast<uint32_t>(std::stoul(arg.substr(13)));
        else if (arg.rfind("--max-upstream-uncertainty-ms=", 0) == 0)
            settings.livekit.max_upstream_uncertainty_ns = static_cast<int64_t>(std::stod(arg.substr(30)) * 1e6);
        else
        {
            usage(argv[0]);
            return arg == "--help" ? 0 : 1;
        }
    }
    if (specs.empty())
    {
        usage(argv[0]);
        return 1;
    }
    bool any_livekit = false;
    for (const StreamSpec& s : specs)
        any_livekit = any_livekit || s.file.empty();
    if (any_livekit && (livekit_url.empty() || token_file.empty()))
    {
        std::cerr << "a --stream with a track needs --livekit-url and --livekit-token-file\n";
        return 1;
    }

    // Only an operation that uses the GPU needs a CUDA context; the default forwards the Orin's access units untouched.
    std::unique_ptr<CudaContext> context;
    if (settings.operation != Operation::None)
    {
        context = std::make_unique<CudaContext>();
        std::cout << "GPU: " << context->device_name() << "\n";
    }
    // One LiveKit participant for every stream, as the Orin producer joins once for all its cameras.
    std::unique_ptr<SharedLiveKit> livekit;
    if (any_livekit)
        livekit = std::make_unique<SharedLiveKit>(livekit_url, read_token(token_file));

    std::vector<std::unique_ptr<StreamWorker>> workers;
    for (const StreamSpec& spec : specs)
        workers.push_back(
            std::make_unique<StreamWorker>(context.get(), spec, settings, spec.file.empty() ? livekit.get() : nullptr));

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    for (auto& w : workers)
        w->start();

    // With --once the streams end by themselves; otherwise run until interrupted.
    std::atomic<bool> finished{ false };
    std::thread waiter(
        [&]
        {
            auto next_stats = std::chrono::steady_clock::now() + std::chrono::seconds(stats_interval_s);
            while (!g_stop && !finished)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (stats_interval_s > 0 && std::chrono::steady_clock::now() >= next_stats)
                {
                    next_stats += std::chrono::seconds(stats_interval_s);
                    for (const auto& w : workers)
                        w->print_live(std::cout);
                    std::cout.flush();
                }
            }
            if (g_stop)
            {
                for (auto& w : workers)
                    w->request_stop();
            }
        });
    bool ok = true;
    for (auto& w : workers)
        ok = w->join() && ok;
    finished = true;
    waiter.join();

    for (const auto& w : workers)
        w->print_summary(std::cout);
    return ok ? 0 : 1;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
