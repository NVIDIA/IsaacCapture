// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuda_context.hpp"

#include <video_stage/nv12.hpp>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

namespace video_stage
{

struct EncoderConfig
{
    Geometry geometry;
    uint32_t fps_numerator = 30;
    uint32_t fps_denominator = 1;
    uint32_t bitrate_bps = 12'000'000;
    /// Frames between IDRs; a stream recovers from loss only at an IDR, so keep it short.
    uint32_t idr_period = 30;
    bool full_range = false;
    /// Reusable input surfaces, registered with the encoder once at startup.
    uint32_t input_surfaces = 3;
};

/**
 * @brief NV12 device surfaces to H.264 access units on NVENC, tuned for low latency.
 *
 * The encoder owns a small pool of CUDA surfaces registered once; the GPU operation writes its output straight
 * into one, so there is no copy and no per-frame registration. Low-latency settings: the ultra-low-latency tuning,
 * no B-frames, no look-ahead, constant bitrate with a one-frame buffer, parameter sets repeated before every IDR.
 * encode() returns the access unit for the picture it was given.
 */
class NvencEncoder
{
public:
    /// A pooled input surface. Write the picture into `view`, then pass the same `slot` to encode().
    struct Surface
    {
        Nv12View view;
        uint32_t slot = 0;
    };

    NvencEncoder(CudaContext& context, const EncoderConfig& config);
    ~NvencEncoder();

    NvencEncoder(const NvencEncoder&) = delete;
    NvencEncoder& operator=(const NvencEncoder&) = delete;

    /// The next free input surface, or nullopt-equivalent `false` when all are in flight.
    bool acquire(Surface& out);

    /// Give back a surface acquired but not encoded (an error path). Idempotent, and encode() releases its own.
    void release(const Surface& surface);

    /// Encode the picture written to `surface` after `stream` has completed the work that wrote it, and release the
    /// surface. `key` comes back on the packet. `force_idr` makes this picture an IDR with parameter sets.
    /// Throws std::runtime_error on an encoder error.
    EncodedPacket encode(const Surface& surface, cudaStream_t stream, uint64_t key, bool force_idr);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace video_stage
