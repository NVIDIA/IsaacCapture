// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Publishes camera_receiver's CUDA IPC streams to a LiveKit room as pre-encoded H.264 tracks.

#include "stream_worker.hpp"

#include <livekit_publisher/livekit_session.hpp>
#include <nvenc/cuda_context.hpp>

#include <atomic>
#include <cctype>
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
           "         [--socket PATH] [--fps N] [--bitrate-mbps N] [--idr-period N] [--gpu N] [--source-full-range]\n         [--timestamp-basis monotonic_us|capture_monotonic_us]\n"
           "         capture_monotonic_us: --chrony-sender HOST[:PORT] --chrony-sender-refid HEX [--chrony-pc-port PORT]\n"
           "           [--chronyc PATH]\n"
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
    ChronyCaptureClockConfig chrony_config;
    bool use_chrony = false;
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
        else if (arg == "--source-full-range")
            defaults.source_full_range = true;
        else if (arg == "--idr-period")
            defaults.idr_period = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (arg == "--timestamp-basis")
        {
            const auto basis = parse_timestamp_basis(next());
            if (!basis)
            {
                std::cerr << "--timestamp-basis must be monotonic_us or capture_monotonic_us\n";
                return 2;
            }
            defaults.timestamp_basis = *basis;
        }
        else if (arg == "--chrony-sender")
        {
            const std::string endpoint = next();
            const auto colon = endpoint.rfind(':');
            chrony_config.sender_host = colon == std::string::npos ? endpoint : endpoint.substr(0, colon);
            if (colon != std::string::npos)
                chrony_config.sender_port = std::atoi(endpoint.substr(colon + 1).c_str());
            if (chrony_config.sender_host.empty())
            {
                std::cerr << "--chrony-sender needs HOST or HOST:PORT, got '" << endpoint << "'\n";
                return 2;
            }
            use_chrony = true;
        }
        else if (arg == "--chrony-sender-refid")
        {
            chrony_config.expected_sender_reference_id = next();
            for (auto& c : chrony_config.expected_sender_reference_id)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        else if (arg == "--chrony-pc-port")
            chrony_config.pc_port = std::atoi(next().c_str());
        else if (arg == "--chronyc")
            chrony_config.chronyc_path = next();
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

    if (defaults.timestamp_basis == TimestampBasis::CaptureMonotonicUs && !use_chrony)
    {
        std::cerr
            << "--timestamp-basis capture_monotonic_us needs --chrony-sender HOST: the sender's capture time is only "
               "used with chrony evidence for both hosts. Use monotonic_us for the frame-ready time.\n";
        return 2;
    }
    if (use_chrony && chrony_config.expected_sender_reference_id.empty())
    {
        std::cerr << "--chrony-sender needs --chrony-sender-refid HEX: the reference id chronyd shows for its source "
                     "(chronyc tracking), so a host following another source is refused.\n";
        return 2;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    try
    {
        CudaContext context(gpu);
        auto session = LiveKitSession::connect(url, read_token(token_file));
        std::unique_ptr<ChronyCaptureClock> chrony_clock;
        if (use_chrony)
        {
            chrony_clock = std::make_unique<ChronyCaptureClock>(chrony_config);
            chrony_clock->poll_once();
            chrony_clock->start();
            defaults.chrony_clock = chrony_clock.get();
        }
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
                              << stats.skipped.load() << " skipped, " << stats.rejected.load() << " rejected, "
                              << stats.slot_overruns.load() << " slot overruns, encode " << mean_ms
                              << " ms mean, bitrate " << stats.bitrate_bps.load() / 1000 << " kbps (transport target "
                              << stats.target_bps.load() / 1000 << " kbps, " << stats.rate_changes.load() << " changes)";
                    const uint64_t ready_samples = stats.capture_to_ready_samples.load();
                    if (ready_samples > 0)
                    {
                        std::cerr << ", capture-to-ready "
                                  << static_cast<double>(stats.capture_to_ready_us_total.load()) / ready_samples / 1000.0
                                  << " ms mean " << stats.capture_to_ready_max_us.load() / 1000.0 << " ms max";
                    }
                    if (defaults.timestamp_basis == TimestampBasis::CaptureMonotonicUs)
                    {
                        std::cerr
                            << ", capture claims: "
                            << stats
                                   .chrony_verdicts[static_cast<std::size_t>(isaaccapture::timesync::Verdict::Qualified)]
                                   .load()
                            << " qualified";
                        for (std::size_t v = 1; v < 10; ++v)
                        {
                            if (stats.chrony_verdicts[v].load() > 0)
                                std::cerr
                                    << ", "
                                    << isaaccapture::timesync::to_string(static_cast<isaaccapture::timesync::Verdict>(v))
                                    << " " << stats.chrony_verdicts[v].load();
                        }
                        std::cerr << ", combined uncertainty " << stats.last_uncertainty_ns.load() / 1000.0
                                  << " us, clock generation " << stats.last_generation.load();
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
