// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mock_nv12_source.cuh"

namespace plugins
{
namespace sensing
{

namespace
{

// Geometry of the cover resize: the source scaled uniformly until it fills the target, then
// centre-cropped. `scaled_*` is the size of the scaled image and `crop_*` the offset of the
// visible window inside it.
struct CoverGeometry
{
    int src_width;
    int src_height;
    int scaled_width;
    int scaled_height;
    int crop_x;
    int crop_y;
};

__device__ float3 fetch_rgb(const uint8_t* src, int pitch, int x, int y)
{
    const uint8_t* px = src + static_cast<size_t>(y) * pitch + static_cast<size_t>(x) * 4;
    return make_float3(px[0], px[1], px[2]);
}

// Colour of one pixel of the cropped output. Enlarging samples bilinearly; reducing averages the
// source area the pixel covers, weighted by overlap, as a box filter would.
__device__ float3 sample_cover(const uint8_t* src, int pitch, const CoverGeometry& g, int x, int y)
{
    const float sx = static_cast<float>(g.src_width) / g.scaled_width;
    const float sy = static_cast<float>(g.src_height) / g.scaled_height;
    const int X = min(max(x + g.crop_x, 0), g.scaled_width - 1);
    const int Y = min(max(y + g.crop_y, 0), g.scaled_height - 1);

    if (sx <= 1.0f && sy <= 1.0f)
    {
        const float fx = fminf(fmaxf((X + 0.5f) * sx - 0.5f, 0.0f), g.src_width - 1.0f);
        const float fy = fminf(fmaxf((Y + 0.5f) * sy - 0.5f, 0.0f), g.src_height - 1.0f);
        const int x0 = static_cast<int>(fx);
        const int y0 = static_cast<int>(fy);
        const int x1 = min(x0 + 1, g.src_width - 1);
        const int y1 = min(y0 + 1, g.src_height - 1);
        const float tx = fx - x0;
        const float ty = fy - y0;
        const float3 a = fetch_rgb(src, pitch, x0, y0);
        const float3 b = fetch_rgb(src, pitch, x1, y0);
        const float3 c = fetch_rgb(src, pitch, x0, y1);
        const float3 d = fetch_rgb(src, pitch, x1, y1);
        return make_float3((a.x * (1 - tx) + b.x * tx) * (1 - ty) + (c.x * (1 - tx) + d.x * tx) * ty,
                           (a.y * (1 - tx) + b.y * tx) * (1 - ty) + (c.y * (1 - tx) + d.y * tx) * ty,
                           (a.z * (1 - tx) + b.z * tx) * (1 - ty) + (c.z * (1 - tx) + d.z * tx) * ty);
    }

    const float left = X * sx;
    const float right = (X + 1) * sx;
    const float top = Y * sy;
    const float bottom = (Y + 1) * sy;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float weight = 0.0f;
    for (int j = static_cast<int>(top); j < min(static_cast<int>(ceilf(bottom)), g.src_height); ++j)
    {
        const float wy = fminf(bottom, j + 1.0f) - fmaxf(top, static_cast<float>(j));
        for (int i = static_cast<int>(left); i < min(static_cast<int>(ceilf(right)), g.src_width); ++i)
        {
            const float w = wy * (fminf(right, i + 1.0f) - fmaxf(left, static_cast<float>(i)));
            const float3 rgb = fetch_rgb(src, pitch, i, j);
            sum.x += rgb.x * w;
            sum.y += rgb.y * w;
            sum.z += rgb.z * w;
            weight += w;
        }
    }
    const float inv = weight > 0.0f ? 1.0f / weight : 0.0f;
    return make_float3(sum.x * inv, sum.y * inv, sum.z * inv);
}

__device__ float luma_709(const float3& rgb)
{
    return fminf(fmaxf(0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z, 0.0f), 255.0f);
}

// One thread per 2x2 block: four luma samples and one chroma pair.
__global__ void paint_gif(const uint8_t* src,
                          int src_pitch,
                          CoverGeometry geometry,
                          uint8_t* y_plane,
                          int y_pitch,
                          uint8_t* uv_plane,
                          int uv_pitch,
                          int width,
                          int height,
                          bool mirror,
                          int shift_x)
{
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= width / 2 || cy >= height / 2)
        return;

    float3 mean = make_float3(0.0f, 0.0f, 0.0f);
    for (int dy = 0; dy < 2; ++dy)
    {
        for (int dx = 0; dx < 2; ++dx)
        {
            const int x = 2 * cx + dx;
            const int y = 2 * cy + dy;
            // A mirrored view samples the picture right to left; the output pixel stays where it is.
            const int sx = (mirror ? width - 1 - x : x) - shift_x;
            const float3 rgb = sample_cover(src, src_pitch, geometry, sx, y);
            y_plane[static_cast<size_t>(y) * y_pitch + x] = static_cast<uint8_t>(luma_709(rgb) + 0.5f);
            mean.x += 0.25f * rgb.x;
            mean.y += 0.25f * rgb.y;
            mean.z += 0.25f * rgb.z;
        }
    }

    // Inverse of the full-range BT.709 matrix the real conversion applies.
    const float luma = luma_709(mean);
    const float cb = fminf(fmaxf(128.0f + (mean.z - luma) / 1.8556f, 0.0f), 255.0f);
    const float cr = fminf(fmaxf(128.0f + (mean.x - luma) / 1.5748f, 0.0f), 255.0f);
    uint8_t* sample = uv_plane + static_cast<size_t>(cy) * uv_pitch + static_cast<size_t>(cx) * 2;
    sample[0] = static_cast<uint8_t>(cb + 0.5f);
    sample[1] = static_cast<uint8_t>(cr + 0.5f);
}

} // namespace

void launch_mock_gif_nv12(const uint8_t* src_rgba,
                          int src_pitch,
                          int src_width,
                          int src_height,
                          uint8_t* y_plane,
                          int y_pitch,
                          uint8_t* uv_plane,
                          int uv_pitch,
                          int width,
                          int height,
                          bool mirror,
                          int shift_x,
                          cudaStream_t stream)
{
    const float scale = fmaxf(static_cast<float>(width) / src_width, static_cast<float>(height) / src_height);
    CoverGeometry geometry{};
    geometry.src_width = src_width;
    geometry.src_height = src_height;
    geometry.scaled_width = max(width, static_cast<int>(lrintf(src_width * scale)));
    geometry.scaled_height = max(height, static_cast<int>(lrintf(src_height * scale)));
    geometry.crop_x = (geometry.scaled_width - width) / 2;
    geometry.crop_y = (geometry.scaled_height - height) / 2;

    const dim3 block(16, 16);
    const dim3 grid((width / 2 + block.x - 1) / block.x, (height / 2 + block.y - 1) / block.y);
    paint_gif<<<grid, block, 0, stream>>>(
        src_rgba, src_pitch, geometry, y_plane, y_pitch, uv_plane, uv_pitch, width, height, mirror, shift_x);
}


} // namespace sensing
} // namespace plugins
