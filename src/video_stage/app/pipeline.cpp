// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <stdexcept>
#include <time.h>

namespace video_stage
{

int64_t monotonic_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

namespace
{
constexpr size_t kMetaRing = 64;
}

struct Pipeline::Impl
{
    CudaContext& context;
    GpuOperation& operation;
    PacketSink& sink;
    PipelineConfig config;
    cudaStream_t stream = nullptr;

    std::array<FrameMeta, kMetaRing> meta{};
    uint64_t next_key = 0;
    bool have_previous_frame = false;
    uint64_t previous_frame = 0;
    bool need_idr = true;

    Geometry input_geometry;
    Geometry output_geometry;
    std::unique_ptr<NvencEncoder> encoder;
    std::unique_ptr<NvdecDecoder> decoder;
    PipelineStats stats;
    int64_t decode_start_ns = 0;

    Impl(CudaContext& ctx, GpuOperation& op, PacketSink& s, const PipelineConfig& cfg)
        : context(ctx), operation(op), sink(s), config(cfg)
    {
        context.make_current();
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("Pipeline: cannot create the CUDA stream");
        decoder = std::make_unique<NvdecDecoder>(
            context, stream, [this](DecodedFrame&& f) { on_frame(std::move(f)); },
            [this](const Geometry& g) { on_format(g); });
    }

    ~Impl()
    {
        decoder.reset();
        encoder.reset();
        context.make_current();
        cudaStreamDestroy(stream);
    }

    // The input geometry changed: prepare the operation and a matching encoder before the next picture. The first
    // output after a change is an IDR.
    void on_format(const Geometry& g)
    {
        input_geometry = g;
        operation.prepare(g);
        output_geometry = operation.output_geometry(g);
        EncoderConfig enc;
        enc.geometry = output_geometry;
        enc.bitrate_bps = config.bitrate_bps;
        enc.idr_period = config.idr_period;
        enc.fps_numerator = config.fps_numerator;
        encoder.reset();
        encoder = std::make_unique<NvencEncoder>(context, enc);
        need_idr = true;
    }

    void on_frame(DecodedFrame&& frame)
    {
        ++stats.pictures;
        const int64_t decoded_ns = monotonic_ns();
        stats.decode_ns.add(decoded_ns - decode_start_ns);
        const FrameMeta& m = meta[frame.key % kMetaRing];

        NvencEncoder::Surface surface;
        if (!encoder || !encoder->acquire(surface))
        {
            ++stats.dropped_no_surface;
            return;
        }
        surface.view.color = frame.view.color;
        operation.enqueue(frame.view, surface.view, stream);
        const bool idr = need_idr;
        need_idr = false;
        EncodedPacket packet = encoder->encode(surface, stream, frame.key, idr);
        frame.lease.reset(); // the encoder has consumed the output; the decoder surface can be reused
        const int64_t done_ns = monotonic_ns();
        stats.process_encode_ns.add(done_ns - decoded_ns);
        stats.receive_to_output_ns.add(done_ns - m.received_ns);
        sink.write(packet, meta[packet.key % kMetaRing], output_geometry);
    }
};

Pipeline::Pipeline(CudaContext& context, GpuOperation& operation, PacketSink& sink, const PipelineConfig& config)
    : m_impl(std::make_unique<Impl>(context, operation, sink, config))
{
}

Pipeline::~Pipeline() = default;

void Pipeline::process(const avs1::Record& record, int64_t received_ns)
{
    Impl& p = *m_impl;
    ++p.stats.records;
    const uint64_t frame = record.sample.frame_number;
    if (p.have_previous_frame && frame != p.previous_frame + 1)
        ++p.stats.gaps;
    p.previous_frame = frame;
    p.have_previous_frame = true;

    const uint64_t key = p.next_key++;
    FrameMeta& slot = p.meta[key % kMetaRing];
    slot.frame_number = frame;
    slot.source_timestamp_ns = record.sample.frame_timestamp_ns;
    slot.received_ns = received_ns;
    p.decode_start_ns = monotonic_ns();
    p.decoder->decode(record.payload, record.payload_size, key);
}

const PipelineStats& Pipeline::stats() const
{
    return m_impl->stats;
}

} // namespace video_stage
