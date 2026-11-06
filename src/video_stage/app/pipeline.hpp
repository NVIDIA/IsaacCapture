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
    int64_t received_ns = 0; ///< CLOCK_MONOTONIC when the intake finished reading the record
    uint64_t key = ~uint64_t{ 0 }; ///< the key this entry was written for; a mismatch means the entry was overwritten
    int64_t submitted_ns = 0; ///< when the access unit was handed to the decoder
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
    /// The source never emits B-frames (the Orin producer's encoder contract). A decoder that cannot see that holds
    /// back pictures to be safe, so the sequence header is rewritten to declare it.
    bool source_has_no_b_frames = true;
    uint32_t bitrate_bps = 12'000'000;
    uint32_t idr_period = 30;
    uint32_t fps_numerator = 30;
};

struct PipelineStats
{
    uint64_t records = 0;
    uint64_t pictures = 0;
    uint64_t dropped_no_surface = 0;
    uint64_t gaps = 0; ///< records whose frame number did not increase
    uint64_t metadata_lost = 0; ///< pictures whose metadata entry had been overwritten
    DurationStats queue_age_ns; ///< from intake to the start of processing
    DurationStats decode_ns; ///< from handing an access unit to the decoder to its picture appearing
    DurationStats decode_lag; ///< records handed to the decoder after this picture's own record, before it appeared
                              ///< (frames)
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

    /// A new source session begins: forget the previous one's frame numbering and start the encoder at an IDR. Call
    /// between sessions, from the thread that calls process().
    void reset_session();

    /// Make the next encoded picture an IDR (a subscriber joined or the receiving side asked). Thread-safe.
    void request_idr();

    const PipelineStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

int64_t monotonic_ns();

} // namespace video_stage
