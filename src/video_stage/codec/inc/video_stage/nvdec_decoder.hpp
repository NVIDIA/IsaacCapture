// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuda_context.hpp"

#include <video_stage/nv12.hpp>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>

namespace video_stage
{

/// What a decoded stream looks like: its size and its quantisation range. A change of either at a sequence header
/// recreates the decoder and is reported to the consumer.
struct StreamFormat
{
    Geometry geometry;
    ColorInfo color;

    bool operator==(const StreamFormat& other) const
    {
        return geometry == other.geometry && color.full_range == other.color.full_range;
    }
};

/// A decoded picture in device memory, still mapped to the decoder's output surface. Keep the lease until the work
/// that reads `view` has completed; releasing it returns the surface to the decoder. `key` is the value given with
/// the access unit that produced the picture.
struct DecodedFrame
{
    Nv12View view;
    uint64_t key = 0;
    std::shared_ptr<void> lease;
};

/**
 * @brief H.264 access units to NV12 device surfaces on NVDEC.
 *
 * decode() feeds one complete Annex-B access unit and returns after the parser has called back. With display delay 0
 * and no B-frames, a picture is delivered from the call that supplied its access unit. Pictures arrive in display
 * order through the callback, on the calling thread. Every picture carries the key of its own access unit, so
 * identity survives any reordering and a lost access unit cannot shift later keys.
 *
 * A change of coded size or format in the stream recreates the decoder at that access unit; on_format_change()
 * tells the consumer, which must reprepare its operation for the new geometry before the next picture.
 */
class NvdecDecoder
{
public:
    using FrameCallback = std::function<void(DecodedFrame&&)>;
    using FormatCallback = std::function<void(const StreamFormat&)>;

    /// `stream` is the CUDA stream the decoded surfaces are made ready on; work enqueued on it afterwards is ordered
    /// after the decode.
    NvdecDecoder(CudaContext& context, cudaStream_t stream, FrameCallback on_frame, FormatCallback on_format_change);
    ~NvdecDecoder();

    NvdecDecoder(const NvdecDecoder&) = delete;
    NvdecDecoder& operator=(const NvdecDecoder&) = delete;

    /// Throws std::runtime_error if the decoder rejects the data.
    void decode(const uint8_t* access_unit, size_t size, uint64_t key);

    /// Pictures delivered so far and access units that produced none; the difference shows decoder-side loss.
    uint64_t pictures() const;
    uint64_t access_units() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace video_stage
