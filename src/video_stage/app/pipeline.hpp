// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "stats.hpp"

#include <video_stage/avs1.hpp>
#include <video_stage/cuda_context.hpp>
#include <video_stage/gpu_operation.hpp>
#include <video_stage/nvdec_decoder.hpp>
#include <video_stage/nvenc_encoder.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace video_stage
{

/// Source identity carried with a picture from the access unit that produced it to the packet that leaves.
struct FrameMeta
{
    uint64_t frame_number = 0;
    uint64_t source_timestamp_ns = 0;
    int64_t received_ns = 0; ///< CLOCK_MONOTONIC when the record was complete on this host
};

/// Where finished access units go. Called on the pipeline thread; it must not block.
class PacketSink
{
public:
    virtual ~PacketSink() = default;
    virtual void write(const EncodedPacket& packet, const FrameMeta& meta, const Geometry& geometry) = 0;
};

struct PipelineConfig
{
    uint32_t bitrate_bps = 12'000'000;
    uint32_t idr_period = 30;
    uint32_t fps_numerator = 30;
};

struct PipelineStats
{
    uint64_t records = 0;
    uint64_t pictures = 0;
    uint64_t dropped_no_surface = 0;
    uint64_t gaps = 0; ///< records whose frame number did not follow the previous one
    DurationStats decode_ns;
    DurationStats process_encode_ns;
    DurationStats receive_to_output_ns;
};

/**
 * @brief Decode -> GPU operation -> encode on one thread, frame by frame.
 *
 * Pictures stay in GPU memory from NVDEC to NVENC. Identity travels by key: each access unit's key indexes a small
 * ring of FrameMeta, and NVDEC and NVENC hand the same key back, so metadata follows its own frame through any loss.
 * A picture arriving while every encoder input surface is busy is dropped and counted, never queued.
 */
class Pipeline
{
public:
    Pipeline(CudaContext& context, GpuOperation& operation, PacketSink& sink, const PipelineConfig& config);
    ~Pipeline();

    /// Process one received record. Throws std::runtime_error on a decoder or encoder failure.
    void process(const avs1::Record& record, int64_t received_ns);

    const PipelineStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

int64_t monotonic_ns();

} // namespace video_stage
