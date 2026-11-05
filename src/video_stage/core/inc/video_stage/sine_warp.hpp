// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_operation.hpp"

namespace video_stage
{

/// Parameters of the example warp. The defaults are a visible but modest ripple.
struct SineWarpParams
{
    /// Peak horizontal displacement in luma pixels. Zero makes the operation an identity.
    float amplitude_px = 8.0F;
    /// Rows per full sine period.
    float wavelength_px = 128.0F;
    /// Phase in radians, applied at row 0.
    float phase_rad = 0.0F;
};

/**
 * @brief EXAMPLE operation: a horizontal sine ripple, not calibrated rectification.
 *
 * Destination-to-source mapping, in luma coordinates:
 *   source_x = x + amplitude * sin(2*pi * y / wavelength + phase),  source_y = y
 * Output pixel (x, y) is the bilinear sample of the input at (source_x, source_y). A source position outside
 * the image gives the border fill (black for the declared range). Output size and color are unchanged. The warped
 * picture no longer has the source camera's calibrated projection.
 *
 * Chroma is sampled from its own plane at the position the same mapping gives for the chroma sample's luma-space
 * location, with the H.264 default siting: chroma sits on even luma columns and half way between luma rows. A
 * replacement operation (the fisheye remap) keeps the same enqueue contract and swaps only the mapping.
 */
class SineWarp final : public GpuOperation
{
public:
    explicit SineWarp(const SineWarpParams& params = {});

    Geometry output_geometry(const Geometry& input) const override;
    void prepare(const Geometry& input) override;
    void enqueue(const Nv12View& in, const Nv12View& out, cudaStream_t stream) override;

private:
    SineWarpParams m_params;
    Geometry m_prepared;
};

} // namespace video_stage
