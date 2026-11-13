// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "box_downscale.hpp"

#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace camera_bridge
{

struct ImageSize
{
    uint32_t width = 0;
    uint32_t height = 0;
    bool operator==(const ImageSize& other) const
    {
        return width == other.width && height == other.height;
    }
};

/**
 * @brief One GPU image operation on an NV12 picture: planes in, planes out, queued on a CUDA stream.
 *
 * The bridge runs a series of steps between the receiver's IPC buffer and the encoder input, all on the GPU, with no
 * host copy. Steps run in the order they are added and each reads the previous step's output. With no steps the
 * picture is only copied. A step is a good place for anything that changes pixels per frame: rectification, a crop,
 * a color change, an overlay.
 *
 * To add a step:
 *   1. Write a class deriving from GpuStep (see SineWarpStep in sine_warp.hpp for a complete, small example).
 *   2. Add one line to make_gpu_step() in gpu_pipeline.cpp so `--gpu-step NAME[=ARGS]` can create it.
 *   3. Pass `--gpu-step NAME=ARGS` once per step, in the order they should run. Nothing runs unless asked for.
 *
 * Contract:
 *  - prepare() runs once per input size, before the first frame. Allocate and upload here (lookup tables,
 *    calibration, scratch). enqueue() must not allocate, synchronise the device or the stream, or throw.
 *  - enqueue() only queues work on `stream` and returns the CUDA status of the launches. The output is not written
 *    when it returns; the caller waits for the stream. `in` and `out` are different buffers (no in-place work).
 *  - Pixels only: frame identity, timestamps and metadata travel with the frame and are not a step's business.
 *  - A step may change the size (output_size()); later steps and the encoder see the new size.
 */
class GpuStep
{
public:
    virtual ~GpuStep() = default;

    virtual const char* name() const = 0;

    /// The size of the picture this step produces from one of `input` size. Default: unchanged.
    virtual ImageSize output_size(const ImageSize& input) const
    {
        return input;
    }

    /// Called once for the size of the picture this step will receive. Returns an empty string when the step can run
    /// on it, otherwise the reason it cannot (for example an odd size, or a calibration made for another size).
    virtual std::string prepare(const ImageSize& input) = 0;

    virtual cudaError_t enqueue(const Nv12Planes& in, const Nv12PlanesOut& out, cudaStream_t stream) = 0;
};

/// A series of steps. Owns the intermediate buffers between them; the last step writes straight into the caller's
/// destination (the encoder input surface).
class GpuPipeline
{
public:
    GpuPipeline() = default;
    ~GpuPipeline();
    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    void add(std::unique_ptr<GpuStep> step);

    /// Prepares every step for a picture of `source` size and allocates the intermediate buffers. Returns an empty
    /// string on success, otherwise which step cannot run and why.
    std::string prepare(const ImageSize& source);

    /// The size of the picture the last step produces (the source size when there are no steps).
    ImageSize output_size() const
    {
        return m_output;
    }

    /// Queues every step on `stream`, source to destination. With no steps, a plain copy.
    cudaError_t enqueue(const Nv12Planes& source, const Nv12PlanesOut& destination, cudaStream_t stream);

private:
    struct Scratch
    {
        uint8_t* y = nullptr;
        uint8_t* uv = nullptr;
        size_t y_pitch = 0;
        size_t uv_pitch = 0;
    };

    std::vector<std::unique_ptr<GpuStep>> m_steps;
    std::vector<ImageSize> m_inputs; // the size each step receives
    std::vector<Scratch> m_scratch; // the output of every step except the last
    ImageSize m_source;
    ImageSize m_output;
};

/**
 * @brief Creates a step from a command-line spec `NAME` or `NAME=ARG[,ARG...]`; throws std::invalid_argument for an
 * unknown name or bad arguments. The single place that maps names to steps: add new steps here.
 */
std::unique_ptr<GpuStep> make_gpu_step(const std::string& spec);

/// A box downscale by an integer divisor, as a step.
std::unique_ptr<GpuStep> make_box_downscale_step(uint32_t divisor);

} // namespace camera_bridge
