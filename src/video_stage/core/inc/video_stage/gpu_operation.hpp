// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nv12.hpp"

#include <cuda_runtime_api.h>

namespace video_stage
{

/**
 * @brief One per-frame GPU image operation: NV12 in, NV12 out, fixed geometry.
 *
 * Contract:
 *  - prepare() runs once per input geometry, outside the frame loop. It may allocate and upload; enqueue() may not.
 *  - enqueue() only queues work on `stream` and returns. It never synchronises the device or the stream, and
 *    allocates nothing. When it returns, the output is not yet written: the caller records a CUDA event on
 *    `stream` after the call, and whoever consumes `out` waits on that event.
 *  - `in` and `out` are distinct surfaces; the operation is not in-place. `in` must stay valid until the work
 *    completes. `out` has output_geometry() and carries the input's color description.
 *  - Frame identity, timestamps and metadata are not an operation's business: they travel with the frame.
 *  - If enqueue() throws, earlier launches of that call may already be in flight on `stream`. The caller drains
 *    the stream (or keeps the surfaces leased) before it recycles `in` or `out`.
 */
class GpuOperation
{
public:
    virtual ~GpuOperation() = default;

    /// The geometry this operation produces for `input`. Constant for a given operation and input.
    virtual Geometry output_geometry(const Geometry& input) const = 0;

    /// Prepare resources for frames of this input geometry. Throws std::invalid_argument if unsupported.
    virtual void prepare(const Geometry& input) = 0;

    /// Queue the operation on `stream`. Throws std::logic_error if not prepared for in.geometry.
    virtual void enqueue(const Nv12View& in, const Nv12View& out, cudaStream_t stream) = 0;
};

} // namespace video_stage
