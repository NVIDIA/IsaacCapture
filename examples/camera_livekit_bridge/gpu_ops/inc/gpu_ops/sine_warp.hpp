// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_step.hpp"

namespace camera_bridge
{

/// Parameters of the example warp. The defaults are a visible but modest ripple.
struct SineWarpParams
{
    /// Peak horizontal displacement in luma pixels. Zero makes the operation an identity.
    float amplitude_px = 8.0F;
    /// Rows per full sine period.
    float wavelength_px = 128.0F;
    /// Phase in radians at row 0.
    float phase_rad = 0.0F;
};

/// True when the parameters are finite with a positive wavelength and the size is a valid NV12 size.
bool sine_warp_supported(uint32_t width, uint32_t height, const SineWarpParams& params);

/**
 * @brief EXAMPLE of a per-frame GPU operation, not calibrated rectification: a horizontal sine ripple.
 *
 * Output pixel (x, y) is the bilinear sample of the source at (x + amplitude * sin(2*pi*y/wavelength + phase), y);
 * a position outside the image gives the video-range black fill. The output has the source's size. Chroma is sampled
 * at the mapped position of its own luma-space location (even columns, half way between rows).
 */
cudaError_t sine_warp_nv12(const Nv12Planes& source,
                           uint32_t width,
                           uint32_t height,
                           const Nv12PlanesOut& destination,
                           const SineWarpParams& params,
                           cudaStream_t stream);

/**
 * @brief The sine warp as a GpuStep: the template to copy for a real step.
 *
 * HOW LENS RECTIFICATION WOULD BE WRITTEN AS A STEP (this class is the same shape, with a different mapping):
 *
 *  - Every pixel of the output picture asks "where in the distorted camera image does this pixel come from?". For
 *    rectification that answer comes from the camera calibration: for the output pixel (u, v) build the ray of the
 *    ideal pinhole camera (the new camera matrix), distort it with the lens model (for example the fisheye or
 *    Brown-Conrady coefficients stored with the camera's intrinsics), project it with the real intrinsics, and you
 *    have the source position (sx, sy). The sine warp replaces exactly this mapping with
 *    `sx = x + A*sin(2*pi*y/L + phase), sy = y`.
 *  - The mapping depends only on the calibration and the picture size, never on the pixels, so do it ONCE:
 *    in prepare(), compute (sx, sy) for every output pixel on the host, cudaMalloc a float2 table of that size
 *    and upload it. Keep the table as a member and free it in the destructor. The per-frame kernel is then only
 *    "read the table entry, bilinear-sample the source" and costs about the same as the sine warp.
 *  - Chroma: the NV12 chroma plane is half size. Either keep a second half-size table, or evaluate the table of
 *    the luma position of each chroma sample ((2*cx, 2*cy + 0.5) here) and halve the result. Samples whose source
 *    position falls outside the image get the fill: video-range black (luma 16, chroma 128).
 *  - Calibration is per camera and per resolution. Give the step its calibration as arguments (a file path, via
 *    make_gpu_step), refuse in prepare() a picture size the calibration was not made for, and remember that a
 *    stream shrunk by `--stream NAME=TRACK:DIV` is shrunk AFTER the steps, so a step always sees the source size.
 *    Rectify at full resolution, then let the downscale shrink the result.
 *  - Rectification usually crops or scales the field of view: that is output_size() (a different size than the
 *    input is allowed; the encoder is created for the last step's output size).
 *  - Several cameras: one pipeline exists per stream, so each stream can get its own calibration; pass the stream
 *    name or a per-stream argument if the steps differ.
 */
class SineWarpStep final : public GpuStep
{
public:
    explicit SineWarpStep(const SineWarpParams& params = {}) : m_params(params)
    {
    }

    const char* name() const override
    {
        return "sine-warp";
    }

    std::string prepare(const ImageSize& input) override
    {
        m_size = input; // remembered: enqueue() gets planes only, the size is known here
        return sine_warp_supported(input.width, input.height, m_params) ?
                   std::string() :
                   "the picture size must be even and the warp parameters finite with a positive wavelength";
    }

    cudaError_t enqueue(const Nv12Planes& in, const Nv12PlanesOut& out, cudaStream_t stream) override
    {
        return sine_warp_nv12(in, m_size.width, m_size.height, out, m_params, stream);
    }

private:
    SineWarpParams m_params;
    ImageSize m_size;
};

} // namespace camera_bridge
