// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"
#include "tcp_receiver.hpp"

#include <video_stage/avs1.hpp>
#include <video_stage/cuda_context.hpp>
#include <video_stage/sine_warp.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

using namespace video_stage;

namespace
{

// Writes the processed access units as one Annex-B stream, so a run can be checked with ffprobe or a player.
class FileSink final : public PacketSink
{
public:
    explicit FileSink(const std::string& path) : m_out(path, std::ios::binary)
    {
        if (!m_out)
            throw std::runtime_error("cannot open " + path);
    }
    void write(const EncodedPacket& packet, const FrameMeta&, const Geometry&) override
    {
        m_out.write(reinterpret_cast<const char*>(packet.data.data()), static_cast<std::streamsize>(packet.data.size()));
    }

private:
    std::ofstream m_out;
};

void print_stage(const char* name, const DurationStats& s)
{
    std::printf("  %-22s n=%zu  p50 %.2f ms  p95 %.2f ms  p99 %.2f ms\n", name, s.count(), s.percentile(50) / 1e6,
                s.percentile(95) / 1e6, s.percentile(99) / 1e6);
}

void usage(const char* program)
{
    std::cout
        << "Usage: " << program << " --port=N --output=FILE.h264 [--amplitude=PX] [--bitrate=BPS] [--idr-period=N]\n"
        << "Listens for one AVS1 (H.264) connection, decodes on NVDEC, applies the example sine warp, re-encodes\n"
        << "on NVENC and writes the access units to FILE. Prints per-stage timing percentiles at the end.\n";
}

} // namespace

int main(int argc, char** argv)
try
{
    uint16_t port = 0;
    std::string output;
    SineWarpParams warp;
    PipelineConfig config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.rfind("--port=", 0) == 0)
            port = static_cast<uint16_t>(std::stoul(arg.substr(7)));
        else if (arg.rfind("--output=", 0) == 0)
            output = arg.substr(9);
        else if (arg.rfind("--amplitude=", 0) == 0)
            warp.amplitude_px = std::stof(arg.substr(12));
        else if (arg.rfind("--bitrate=", 0) == 0)
            config.bitrate_bps = static_cast<uint32_t>(std::stoul(arg.substr(10)));
        else if (arg.rfind("--idr-period=", 0) == 0)
            config.idr_period = static_cast<uint32_t>(std::stoul(arg.substr(13)));
        else
        {
            usage(argv[0]);
            return arg == "--help" ? 0 : 1;
        }
    }
    if (port == 0 || output.empty())
    {
        usage(argv[0]);
        return 1;
    }

    CudaContext context;
    std::cout << "GPU: " << context.device_name() << "\n";
    SineWarp operation(warp);
    FileSink sink(output);
    Pipeline pipeline(context, operation, sink, config);
    TcpReceiver receiver(port);
    std::cout << "listening on port " << port << std::endl;
    receiver.accept_connection();

    std::vector<uint8_t> bytes;
    while (receiver.read_record(bytes))
    {
        const int64_t received_ns = monotonic_ns();
        std::string error;
        const auto record = avs1::parse(bytes.data(), bytes.size(), error);
        if (!record)
            throw std::runtime_error("bad AVS1 record: " + error);
        pipeline.process(*record, received_ns);
    }

    const PipelineStats& s = pipeline.stats();
    std::printf("records %llu, pictures %llu, dropped (no surface) %llu, frame-number gaps %llu\n",
                static_cast<unsigned long long>(s.records), static_cast<unsigned long long>(s.pictures),
                static_cast<unsigned long long>(s.dropped_no_surface), static_cast<unsigned long long>(s.gaps));
    print_stage("decode", s.decode_ns);
    print_stage("warp + encode", s.process_encode_ns);
    print_stage("receive -> output", s.receive_to_output_ns);
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
