// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A camera producer without cameras. It runs the same ControlServer, ControlHandler, ProducerSupervisor and
// DirectSender as camera_plugin_sensing and differs only in where frames come from: H.264 files looped at a fixed rate.
//
//   sensing_mock_producer --control-token-file=token.txt --stream=sensing-left=left.h264,direct=127.0.0.1:5600
//   --stream=sensing-right=right.h264
//
// --simulate-fault-seconds=N makes the first N seconds behave like a camera with no power, so the fault reporting and
// the retry can be seen without unplugging anything.

#include <sensing_control/control_handler.hpp>
#include <sensing_control/control_server.hpp>
#include <sensing_control/producer_supervisor.hpp>
#include <video_stage/direct_sender.hpp>
#include <video_stage/h264_access_units.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace plugins::sensing;

namespace
{

std::atomic<bool> g_stop{ false };

void on_signal(int)
{
    g_stop = true;
}

struct StreamArgs
{
    std::string track;
    std::string file;
    std::string direct_host;
    std::uint16_t direct_port = 0;
};

/// "TRACK=FILE[,direct=HOST:PORT]"
StreamArgs parse_stream(const std::string& text)
{
    StreamArgs out;
    const auto eq = text.find('=');
    if (eq == std::string::npos || eq == 0)
        throw std::runtime_error("--stream must look like TRACK=FILE[,direct=HOST:PORT]: " + text);
    out.track = text.substr(0, eq);
    std::string rest = text.substr(eq + 1);
    const auto comma = rest.find(",direct=");
    if (comma != std::string::npos)
    {
        const std::string target = rest.substr(comma + 8);
        rest = rest.substr(0, comma);
        const auto colon = target.rfind(':');
        if (colon == std::string::npos)
            throw std::runtime_error("direct= must be HOST:PORT: " + target);
        out.direct_host = target.substr(0, colon);
        out.direct_port = static_cast<std::uint16_t>(std::stoul(target.substr(colon + 1)));
    }
    out.file = rest;
    return out;
}

std::uint64_t monotonic_us()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

/// One looped file and the direct sender that carries it.
struct MockStream
{
    std::string track;
    std::vector<std::uint8_t> bytes;
    std::vector<video_stage::AccessUnitRange> units;
    std::size_t cursor = 0;
    std::uint32_t frame_id = 0;
    std::uint64_t frames = 0;
    std::unique_ptr<DirectSender> sender;
};

class MockSession : public CameraSession
{
public:
    MockSession(const std::vector<StreamArgs>& streams, const DirectTargets& targets, std::uint32_t fps)
        : m_period(std::chrono::microseconds(1'000'000 / fps)), m_next(std::chrono::steady_clock::now())
    {
        for (const auto& args : streams)
        {
            auto stream = std::make_unique<MockStream>();
            stream->track = args.track;
            std::ifstream file(args.file, std::ios::binary);
            if (!file)
                throw std::runtime_error("cannot read " + args.file);
            stream->bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            stream->units = video_stage::split_access_units(stream->bytes.data(), stream->bytes.size());
            if (stream->units.empty())
                throw std::runtime_error(args.file + " has no H.264 access units");

            DirectSenderConfig config;
            config.fps = fps;
            const auto found = targets.find(args.track);
            if (found != targets.end())
            {
                config.host = found->second.host;
                config.port = found->second.port;
            }
            stream->sender = std::make_unique<DirectSender>(config);
            m_streams.push_back(std::move(stream));
        }
    }

    void update() override
    {
        const auto now = std::chrono::steady_clock::now();
        while (now >= m_next)
        {
            m_next += m_period;
            for (auto& stream : m_streams)
                emit(*stream);
        }
    }

    void flush() override
    {
    }

    std::vector<StreamStatus> streams() const override
    {
        std::vector<StreamStatus> out;
        for (const auto& stream : m_streams)
        {
            StreamStatus entry;
            entry.track = stream->track;
            entry.frames = stream->frames;
            entry.published = stream->frames;
            entry.direct_host = stream->sender->host();
            entry.direct_port = stream->sender->port();
            entry.direct_connected = stream->sender->connected();
            entry.direct_error = stream->sender->last_error();
            out.push_back(std::move(entry));
        }
        return out;
    }

    void set_direct_target(const std::string& track, const std::string& host, std::uint16_t port) override
    {
        for (auto& stream : m_streams)
            if (stream->track == track)
                stream->sender->set_target(host, port);
    }

private:
    void emit(MockStream& stream)
    {
        // A keyframe request is answered at the file's next IDR, as an encoder would answer it.
        if (stream.sender->take_keyframe_request())
        {
            for (std::size_t k = 0; k < stream.units.size(); ++k)
            {
                const std::size_t index = (stream.cursor + k) % stream.units.size();
                if (stream.units[index].idr)
                {
                    stream.cursor = index;
                    break;
                }
            }
        }
        const auto& unit = stream.units[stream.cursor];
        stream.cursor = (stream.cursor + 1) % stream.units.size();

        EncodedVideoFrame frame;
        frame.data = stream.bytes.data() + unit.offset;
        frame.size = unit.size;
        frame.keyframe = unit.idr;
        frame.timestamp_us = monotonic_us();
        frame.user_timestamp_us = frame.timestamp_us;
        frame.frame_id = ++stream.frame_id;
        stream.sender->submit(frame);
        ++stream.frames;
    }

    std::chrono::microseconds m_period;
    std::chrono::steady_clock::time_point m_next;
    std::vector<std::unique_ptr<MockStream>> m_streams;
};

void print_usage(const char* program)
{
    std::cout << "Usage: " << program << " --control-token-file=PATH --stream=TRACK=FILE[,direct=HOST:PORT] ...\n"
              << "  --control-port=N          management channel port (default: 5610; 0 picks a free one)\n"
              << "  --control-bind=ADDR       IPv4 address to listen on (default: 127.0.0.1)\n"
              << "  --fps=N                   frames per second per stream (default: 30)\n"
              << "  --retry-seconds=N         wait between start attempts (default: 2)\n"
              << "  --simulate-fault-seconds=N  behave like an unpowered camera for the first N seconds\n"
              << "  --release=ID              release id to report (default: mock)\n";
}

} // namespace

int main(int argc, char** argv)
try
{
    std::vector<StreamArgs> streams;
    ControlServerConfig control;
    control.bind_address = "127.0.0.1";
    std::string token_file;
    std::uint32_t fps = 30;
    std::uint32_t retry_seconds = 2;
    std::uint32_t fault_seconds = 0;
    std::string release = "mock";

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto value = [&](const char* name) { return arg.substr(std::string(name).size()); };
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg.rfind("--stream=", 0) == 0)
            streams.push_back(parse_stream(value("--stream=")));
        else if (arg.rfind("--control-token-file=", 0) == 0)
            token_file = value("--control-token-file=");
        else if (arg.rfind("--control-port=", 0) == 0)
            control.port = static_cast<std::uint16_t>(std::stoul(value("--control-port=")));
        else if (arg.rfind("--control-bind=", 0) == 0)
            control.bind_address = value("--control-bind=");
        else if (arg.rfind("--fps=", 0) == 0)
            fps = static_cast<std::uint32_t>(std::stoul(value("--fps=")));
        else if (arg.rfind("--retry-seconds=", 0) == 0)
            retry_seconds = static_cast<std::uint32_t>(std::stoul(value("--retry-seconds=")));
        else if (arg.rfind("--simulate-fault-seconds=", 0) == 0)
            fault_seconds = static_cast<std::uint32_t>(std::stoul(value("--simulate-fault-seconds=")));
        else if (arg.rfind("--release=", 0) == 0)
            release = value("--release=");
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    if (streams.empty() || token_file.empty() || fps == 0)
    {
        print_usage(argv[0]);
        return 1;
    }
    control.token = read_secret_file(token_file);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    ProducerStatus status(release);
    std::vector<StreamStatus> configured;
    for (const auto& args : streams)
    {
        StreamStatus entry;
        entry.track = args.track;
        entry.direct_host = args.direct_host;
        entry.direct_port = args.direct_port;
        configured.push_back(entry);
    }

    const auto started = std::chrono::steady_clock::now();
    const SessionFactory factory = [&](const DirectTargets& targets) -> std::unique_ptr<CameraSession>
    {
        if (std::chrono::steady_clock::now() - started < std::chrono::seconds(fault_seconds))
            throw std::runtime_error("INvSIPLCamera::Init failed with SIPLStatus 10");
        return std::make_unique<MockSession>(streams, targets, fps);
    };

    std::unique_ptr<ControlServer> server;
    SupervisorConfig supervisor_config;
    supervisor_config.retry_interval = std::chrono::seconds(retry_seconds);
    supervisor_config.poll_interval = std::chrono::milliseconds(2);
    ProducerSupervisor supervisor(
        status, configured, factory,
        [&server](const std::string& text)
        {
            if (server)
                server->broadcast(text);
        },
        supervisor_config);
    ControlHandler handler(status, supervisor);
    server =
        std::make_unique<ControlServer>(control, [&handler](const std::string& text) { return handler.handle(text); });
    std::cout << "Mock producer: management channel on " << control.bind_address << ":" << server->port() << std::endl;

    supervisor.run([] { return g_stop.load(); });
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
